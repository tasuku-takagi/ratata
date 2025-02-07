import argparse 

import cv2
import time

import torch

import sys
sys.path.append("../")
from ratata.inference.video_inference import VideoInferenceYolo
from ratata.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config.yml")
    parser.add_argument("cam_id", help="cam_id")
    return parser.parse_args()

def main(cam_id, device, model_path, model_path_tie, 
         target_size, blur, matt, enhance,debug):
    # VideoInferenceインスタンスを作成
    video_inference = VideoInferenceYolo(model_path, target_size, device=device, hflip=True,
                                         blur=blur, model_path_tie=model_path_tie, debug=debug,
                                         matt=matt, enhance=enhance)

    # Webカメラのキャプチャを開始
    cap = cv2.VideoCapture(cam_id)  # デフォルトカメラ(0)を使用
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # 推論の実行
        start_time = time.time()
        # print(frame.shape)
        binary_mask, result = video_inference.inference_frame(frame)
        # print(binary_mask.shape)
        elapsed_time = time.time() - start_time

        elapsed_time_ms = elapsed_time * 1000
        print(f"iou: {result[0]}")
        print(f"Elapsed time: {elapsed_time_ms:.2f} ms")

        # 結果（バイナリマスク）を表示
        cv2.imshow('Binary Mask', binary_mask)
        # cv2.imshow('Binary_mask', trimap)

        # 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # キャプチャを終了し、ウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config_path)
    
    main(int(args.cam_id), device, config.model_path, config.model_path_tie, 
         config.target_size, config.blur, config.matt, config.enhance, config.debug)
