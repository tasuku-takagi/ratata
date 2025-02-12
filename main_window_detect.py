import argparse

import cv2
import numpy as np
import time

import torch

import sys
sys.path.append("../")
from ratata.inference.video_inference import VideoInferenceYolo
from ratata.production import SrcSelector
from ratata.config import load_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config.yml")
    parser.add_argument("cam_id", help="cam_id")
    parser.add_argument("--offset_y", type=int, default=0, help="Offset in y direction for mask blending")
    return parser.parse_args()


def is_person_detected_by_ratio(mask, ratio_threshold=0.1):
    """
    マスク全体における「0以外ピクセルの割合」が ratio_threshold を超えるかどうか。
    - mask: 0 or 255 の二値画像 (グレースケール)
    - ratio_threshold: この割合を超えたら「人がいる」とみなす (0.0～1.0)
    """
    # 不透明(255)ピクセル数
    non_zero_count = np.count_nonzero(mask)
    total_count = mask.size  # マスク全体のピクセル数

    ratio = non_zero_count / float(total_count)
    return ratio >= ratio_threshold


def create_silhouette_alpha(gray_mask, offset_x, offset_y, bg, silhouette_opacity=1.0):
    """
    グレースケール値(0～255)を 0.0～1.0 のアルファ値として扱い、
    画素ごとにブレンドして合成する。
    加えて、silhouette_opacity (0.0～1.0) をかけることで
    シルエット全体の透明度をコントロールする。
    """
    bg_h, bg_w = bg.shape[:2]
    out_img = bg.copy()

    mask_h, mask_w = gray_mask.shape[:2]
    x1, y1 = offset_x, offset_y
    x2, y2 = x1 + mask_w, y1 + mask_h

    x1_clamped = max(0, min(x1, bg_w))
    y1_clamped = max(0, min(y1, bg_h))
    x2_clamped = max(0, min(x2, bg_w))
    y2_clamped = max(0, min(y2, bg_h))
    roi_w = x2_clamped - x1_clamped
    roi_h = y2_clamped - y1_clamped
    if roi_w <= 0 or roi_h <= 0:
        return out_img

    bg_roi = out_img[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
    mask_roi = gray_mask[0:roi_h, 0:roi_w]

    # マスクを0～1に正規化
    alpha_roi = mask_roi.astype(np.float32) / 255.0  # shape=(roi_h, roi_w)
    # シルエット全体の不透明度を乗じる (全体のフェード)
    alpha_roi *= silhouette_opacity

    # シルエットを白(255,255,255)で表現
    fg_roi = np.ones_like(bg_roi, dtype=np.float32) * 255.0  # (roi_h, roi_w, 3)

    # alpha_roi は (roi_h, roi_w) なので、(roi_h, roi_w, 1) に変形
    alpha_3d = alpha_roi.reshape(roi_h, roi_w, 1)

    # ブレンド計算 (アルファ合成)
    blended_roi = alpha_3d * fg_roi + (1.0 - alpha_3d) * bg_roi.astype(np.float32)
    blended_roi = blended_roi.astype(np.uint8)

    out_img[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = blended_roi
    return out_img


def main(cam_id, device, model_path, model_path_tie, 
         target_size, blur, matt, enhance, debug, 
         fade_in_end, hold_end, fade_out_end, ratio_threshold, 
         offset_x, offset_y,
         bg_dir, bg_names, video_dir, video_names,
         window):
    
    """first sample"""
    src_selector = SrcSelector(bg_dir, bg_names, video_dir, video_names)
    bg_path, video_path = src_selector.get_next_src()
    bg = cv2.imread(bg_path)

    if window:
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        bg = cv2.resize(bg, [target_size[0], target_size[1]], cv2.INTER_AREA)

    STATE_WAITING = 0
    STATE_DISPLAY_SILHOUETTE = 1
    STATE_PLAY_VIDEO = 2
    current_state = STATE_WAITING

    # VideoCapture
    cap_video = cv2.VideoCapture(video_path)
    if not cap_video.isOpened():
        raise FileNotFoundError("動画ファイルが見つからないか、読み込みに失敗しました: " + video_path)

    video_inference = VideoInferenceYolo(model_path, target_size, device=device, hflip=True,
                                         blur=blur, model_path_tie=model_path_tie, debug=debug,
                                         matt=matt, enhance=enhance)
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        """共通処理"""
        print(f"current_state: {current_state}")
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        """state分岐"""
        if current_state == STATE_WAITING: # 推論し続けるが、人が検出されない状態
            # 背景を表示して待機
            cv2.imshow("window", bg)
            mask, _ = video_inference.inference_frame(frame)
            is_detected = is_person_detected_by_ratio(mask, ratio_threshold)
            # 一度 is_detected が True になると、画面合成フェーズへ遷移(次frameから)
            print(f"is_detected: {is_detected}")
            if is_detected:
                current_state = STATE_DISPLAY_SILHOUETTE
                start = time.time()
        elif current_state == STATE_DISPLAY_SILHOUETTE:
            mask, _ = video_inference.inference_frame(frame)
            is_detected = is_person_detected_by_ratio(mask, ratio_threshold)
            print(f"is_detected: {is_detected}")
            if not is_detected: # 検出されなかったら戻る
                current_state = STATE_WAITING
                alpha = 0  # bgのみ映す
            else:
                now = time.time()
                center_hold_start_time = now - start
                # 経過時間に合わせてalphaを設定 → 一定時間で動画に移行
                if center_hold_start_time < fade_in_end:
                    # フェードイン (0→1)
                    alpha = center_hold_start_time / fade_in_end
                elif center_hold_start_time < hold_end:
                    # 完全表示 (1.0)
                    alpha = 1.0
                elif center_hold_start_time < fade_out_end:
                    # フェードアウト (1.0→0.0)
                    alpha = 1.0 - (center_hold_start_time - hold_end) / (fade_out_end - hold_end)
                else:
                    # フェードアウト完了 → 動画再生へ
                    current_state = STATE_PLAY_VIDEO
                    cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)   # videoのframeのみ設定
            # nowのシルエットを描画する
            ## 背景サイズに合わせてマスクをリサイズ
            bg_h, bg_w = bg.shape[:2]
            resized_mask = cv2.resize(mask, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
            # シルエット合成
            silhouette_img = create_silhouette_alpha(
                resized_mask,
                offset_x=offset_x,
                offset_y=offset_y,
                bg=bg,
                silhouette_opacity=alpha
            )
            cv2.imshow("window", silhouette_img)
        elif current_state == STATE_PLAY_VIDEO:
            # 動画再生
            ret, video_frame = cap_video.read()
            if not ret:
                # 動画終了したらWAITINGへ
                current_state = STATE_WAITING
                """videoとbgを次のものに変える"""
                cap_video.release()
                bg_path, video_path = src_selector.get_next_src()
                bg = cv2.imread(bg_path)
                cap_video = cv2.VideoCapture(video_path)
                continue    # 何も移さずに次のframeへ

            # 背景と同じサイズにリサイズして表示
            video_frame_resized = cv2.resize(video_frame, (bg.shape[1], bg.shape[0]))
            cv2.imshow("window", video_frame_resized)

        # TODO: 後で実装
        # elif current_state == STATE_AFTER_VIDEO: 
        #     # ビデオ表示後、背景を動画の最終フレームとし、背景として表示。またこの間はmaskを合成しない。
        #     # 一定時間待ったら背景が徐々に"background.jpg"が不透明度が上がっていき、不透明度が100%になったらcurrent_stateを0に戻す

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cap_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config_path)
    main(int(args.cam_id), device, config.model_path, config.model_path_tie, 
         config.target_size, config.blur, config.matt, config.enhance, config.debug,
         config.fade_in_end, config.hold_end, config.fade_out_end, config.ratio_threshold,
         config.offset_x, args.offset_y,
         config.bg_dir, config.bg_names, config.video_dir, config.video_names,
         config.window)
    