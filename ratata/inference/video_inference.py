from ultralytics import YOLO
from transformers import VitMatteImageProcessor, VitMatteForImageMatting
from huggingface_hub import hf_hub_download

from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image

class VideoInferenceYolo:
    def __init__(self, model_path, target_size, device=None, hflip=False, 
                 blur = "", model_path_tie=None, debug=False,
                 matt=False, enhance=False):
        """
        Args:
            target_size(typle or list): [w, h]
            blur: GussianBlur or DistTransform or None
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.hflip = hflip
        self.target_size = target_size
        self.model = YOLO(model_path)
        self.prev_prediction = None
        self.iou_thr = 0.3
        self.conf_thr = 0.8
        self.blur = blur
        # tie
        self.model_path_tie = model_path_tie
        if self.model_path_tie:
            self.model_tie = YOLO(model_path_tie)
        else:
            self.model_tie = None
        self.tie_no_detection_count = 0
        self.tie_frame_thr = 20
        self.prev_prediction_tie = None
        # 全クラス認識
        self.debug = debug
        # matting
        self.matt = matt
        self.model_matt = None
        self.processor_matt = None
        self.enhance=enhance
        if self.matt:
            self.init_vitmatte()
    
    def init_vitmatte(self, model_name="hustvl/vitmatte-small-composition-1k"):
        self.processor_matt = VitMatteImageProcessor.from_pretrained(model_name)
        self.model_matt = VitMatteForImageMatting.from_pretrained(model_name)
        self.model_matt.to(self.device)

    def inference_frame(self, frame):
        """
        Args:
            frame(np.ndarray): [h, w, c] BGR
        Return:
            output(np.ndarray): [h, w] np.uint8
        """
        # inference
        if frame is None:
            output = np.zeros((self.target_size[1], self.target_size[0])).astype(np.uint8)
            return output, (-1, output)
        if not self.debug:
            results = self.model.predict(frame,
                                        half=True,
                                        device=self.device,
                                        max_det=3,
                                        classes = [0],
            )
        else:
            results = self.model.predict(frame,
                                         half=True,
                                         device=self.device,
                                        #  imgsz = self.target_size,
                                         )
        # 検出なしならゼロ絵を返す
        if results[0].masks is None:
            output = np.zeros((self.target_size[1], self.target_size[0])).astype(np.uint8)
            return output, (-1, output)
        masks = results[0].masks.data   # torch.Tensor([ch, h, w])
        confs = results[0].boxes.conf   # torch.Tensor([ch])
        # 直近の被写体に近いmaskを探す
        output, iou = self.update_prev_prediction(masks, confs)
        # tieをpersonに合体
        if self.model_tie is not None:
            output = self.combine_tie(frame, output)
        # matt
        trimap = None
        if self.matt:
            output, trimap = self.get_matt(frame, output)
            if self.enhance:
                output = (output * 2.0)
                output = torch.clamp(output, 0.0, 1.0)
        # 後処理
        output = output.cpu().numpy() * 255
        output = output.astype(np.uint8)
        # blur
        if self.blur == "GaussianBlur":
            print("GussianBlur")
            output = self.gaussian_blur(output)
        elif self.blur == "DistTransform":
            output == self.dist_transform(output)
        elif self.blur == "soft":
            print("soft")
            output = self.soft_blur(output)
        output = cv2.resize(output, self.target_size, interpolation=cv2.INTER_AREA)
        if self.hflip:
            output = output[:, ::-1]
        return output.astype(np.uint8), (iou, trimap)
    
    def update_prev_prediction(self, masks, confs):
        # prev_predictionに近いmaskをmasksから選択する
        if self.prev_prediction is None:    # 初期状態 → 中央付近のmasksを選択
            output = self.select_mask_based_on_centroid(masks, confs)
            iou = 1
        else:   # prev_maskに最も近いmaskを探す
            output, iou = self.find_most_similar_iou(self.prev_prediction, masks)
        # prev_predictionの更新
        if iou > self.iou_thr:   # 追っていた被写体がまだいる判定
            self.prev_prediction = output
        else:       # 追っていた被写体はいなくなった
            self.prev_prediction = None
        return output,  iou
    
    def select_mask_based_on_centroid(self, masks, confs):
        valid_channels = confs > self.conf_thr
        # confがconf_thrを超えるindexが2つ以上ある場合
        if valid_channels.sum() > 1:
            # 中央座標
            h, w = masks.shape[1], masks.shape[2]
            center = torch.tensor([h / 2, w / 2])
            # valid_channelsに対応するマスクのcentroidを計算
            centroids = []
            for ch in range(masks.shape[0]):
                if valid_channels[ch]:
                    mask = masks[ch, :, :]
                    centroid = self.compute_centroid(mask)
                    centroids.append((ch, centroid))
            # 画像の中央に最も近いcentroidを持つchを選択
            selected_ch, _ = min(centroids, key=lambda x: torch.dist(x[1], center))
            # 選択されたchのmaskを返す
            return masks[selected_ch, :, :]
        else:
            # confが閾値を超えるchが複数無い場合は、最大のconfを持つmaskを返す
            max_conf_ch = torch.argmax(confs)
            return masks[max_conf_ch, :, :]
    
    def get_matt(self, frame, mask):
        trimap = (self.get_trimap(mask) * 255).astype(np.uint8)

        frame = cv2.resize(frame, (trimap.shape[1], trimap.shape[0]), cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor_matt(images=frame, trimaps=trimap, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            alphas = self.model_matt(**inputs).alphas
        return alphas[0, 0, :], trimap
    
    def get_trimap(self, mask, dilation_size=5, dilation_iterations=5):
        trimap = torch.zeros_like(mask, dtype=torch.float32)
        trimap[mask == 1] = 1
        trimap[mask == 0] = 0
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        mask_np = mask.cpu().numpy().astype(np.uint8)
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=dilation_iterations)
        eroded_mask = cv2.erode(mask_np, kernel, iterations=dilation_iterations)
        dilated_mask = torch.from_numpy(dilated_mask).to(self.device)
        eroded_mask = torch.from_numpy(eroded_mask).to(self.device)
        trimap[(dilated_mask == 1) & (eroded_mask == 0)] = 0.5
        return trimap.cpu().numpy()

    
    def gaussian_blur(self, mask):
        # thr=(100, 200): 低い値 → ぼかしをかけるedgeが多く採用される
        edges = cv2.Canny(mask, 100, 200)
        # kernel_size=(5, 5) → edgeが広くぼける
        # iteration=2: 回数が大きいと広くぼける
        dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        # kernel_size=(15, 15) → edgeが広くぼける
        # sigmaX=5: ガウシアン分布の広がり
        blurred = cv2.GaussianBlur(mask, (5, 5), 2)
        result=  np.where(dilated > 0, blurred, mask)
        return result
    
    def soft_blur(self, mask):
        # thr=(100, 200): 低い値 → ぼかしをかけるedgeが多く採用される
        edges = cv2.Canny(mask, 100, 200)
        # kernel_size=(5, 5) → edgeが広くぼける
        # iteration=2: 回数が大きいと広くぼける
        dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        # kernel_size=(15, 15) → edgeが広くぼける
        # sigmaX=5: ガウシアン分布の広がり
        blurred = cv2.GaussianBlur(mask, (5, 5), 1)
        result=  np.where(dilated > 0, blurred, mask)
        return result
    
    def dist_transform(self, mask):
        dist_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(mask, (15, 15), 5)
        result = (mask * (1 - dist_transform) + blurred * dist_transform).astype(np.uint8)
        return result

    def iou(self, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """2つのbinary-maskのiouを返す"""
        intersection = torch.logical_and(mask1, mask2).sum().float()
        union = torch.logical_or(mask1, mask2).sum().float()
        return intersection / union if union > 0 else torch.tensor(0.0)

    def find_most_similar_iou(self, target_mask: torch.Tensor, candidate_masks: torch.Tensor) -> torch.Tensor:
        """target_maskに対して最もiouがmaskをcandidate_masksから選ぶ.
            各maskはbinary-mask.

        target_mask: [h, w]
        candidate_masks: [ch, h, w]
        """
        ch = candidate_masks.shape[0]
        ious = torch.tensor([self.iou(target_mask, candidate_masks[i]) for i in range(ch)])

        max_index = ious.argmax().item()
        return candidate_masks[max_index], ious[max_index]
    
    def combine_tie(self, frame, mask_person):
        mask_tie = self.inference_tie(frame, mask_person)
        out = mask_person.bool() | mask_tie.bool()
        return out

    def inference_tie(self, frame, mask_person):
        results_tie = self.model_tie.predict(frame,
                                             half=True,
                                             device=self.device,
                                             max_det=2,
                                             classes = [27])
        if results_tie[0].masks is None:
            # tieの認識無し
            self.tie_no_detection_count += 1
            if self.tie_no_detection_count < self.tie_no_detection_count:
                if self.prev_prediction_tie is None:
                    self.prev_prediction_tie = torch.zeros(mask_person.shape).to(self.device)
            else:
                self.prev_prediction_tie = torch.zeros(mask_person.shape).to(self.device)
                self.tie_no_detection_count = self.tie_no_detection_count # fail-safe
            print(self.tie_no_detection_count)
            return self.prev_prediction_tie
        masks_tie = results_tie[0].masks.data
        closest_mask_tie, _ = self.find_closest_tie(mask_person, masks_tie)
        self.prev_prediction_tie = closest_mask_tie # 最新のtie-maskを更新
        self.tie_no_detection_count = 0
        print(self.tie_no_detection_count)
        return closest_mask_tie

    def compute_centroid(self, mask):
        y, x = torch.where(mask)
        return torch.tensor([y.float().mean(), x.float().mean()])
    
    def find_closest_tie(self, mask_person, masks_tie):
        person_centroid = self.compute_centroid(mask_person)
        ch = masks_tie.shape[0]
        centroids = torch.stack([self.compute_centroid(masks_tie[i]) for i in range(ch)])
        distances = torch.norm(centroids - person_centroid, dim=1)
        min_index = distances.argmin().item()
        return masks_tie[min_index], distances[min_index]

    
    def inference_video(self, video_path, out_dir):
        # input
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = self.target_size[0] if self.target_size else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = self.target_size[1] if self.target_size else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # output
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_video_name = f"{Path(video_path).stem}_mask"
        out_video_path = Path(out_dir).joinpath(out_video_name).with_suffix(".mp4")
        out_video = cv2.VideoWriter(filename=str(out_video_path),
                                    fourcc=fourcc,
                                    fps=fps,
                                    frameSize=(int(width), int(height)),
                                    isColor=True)
        # inference
        for idx in range(total_frames): 
            ret, frame = cap.read()
            if not ret:
                break
            mask, _ = self.inference_frame(frame)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # [h, w] -> [h, w, 3]
            out_video.write(mask)
        cap.release()
        out_video.release()
        print(f"total frame: {idx}")


    


class VideoInference:
    def __init__(self, model_path, target_size, device=None, quantize=False, slog3=False):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"device: {self.device}")
        if quantize:
            self.model = self.load_quantized_model(model_path)
        else:
            self.model = self.load_model(model_path)
        self.target_size = target_size
        self.slog3 = slog3
        self.brack_frame = self.create_black_frame(self.target_size)
    
    def load_model(self, model_path):
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model

    def load_quantized_model(self, model_path):
        quantized_model = self.load_model(model_path)
        # 量子化された状態で読み込む
        quantized_model.qconfig = torch.quantization.get_default_qconfig('x86')
        torch.quantization.prepare(quantized_model, inplace=True)
        torch.quantization.convert(quantized_model, inplace=True)
        return quantized_model

    def create_black_frame(self, target_size):
        width, height = target_size
        black_frame = np.zeros((height, width), dtype=np.float32)  # [H, W]
        black_tensor = torch.tensor(black_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, H, W]
        return black_tensor
    
    def slog3_to_rec709(self, img):
        # S-Log3からリニアに戻す（適切な係数を使用）
        img_linear = np.power(img / 128.0, 2.4) * 255  # S-Log3のデータ範囲に合わせる
        img_linear = np.clip(img_linear, 0, 255).astype(np.uint8)
        # Rec.709変換 (ガンマ補正)
        img_rec709 = np.clip(img_linear, 0, 255).astype(np.uint8)
        
        return img_rec709
    
    def preprocess_frame(self, frame):
        image = cv2.resize(frame, self.target_size)
        if self.slog3:
            image = self.slog3_to_rec709(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_array = np.array(image).astype(np.float32) / 255.0
        # ImageNetの平均と標準偏差で正規化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std

        image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        return image_tensor
        
    def inference(self, input_tensor, prev_frame_tensor):
        with torch.no_grad():
            output = self.model(input_tensor, prev_frame_tensor)
            output_np = output.squeeze().cpu().numpy()
            binary_mask = (output_np * 255).astype(np.uint8)
        return binary_mask

    def inference_frame(self, frame):
        input_tensor = self.preprocess_frame(frame)
        binary_mask = self.inference(input_tensor, self.brack_frame)
        return binary_mask
