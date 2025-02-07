import cv2
import numpy as np
import time

import sys
sys.path.append("../")
from ratata.inference.video_inference import VideoInference


def is_person_in_center(mask, rect_w, rect_h):
    """
    画面中央に縦長の矩形 (幅=rect_w, 高さ=rect_h) を想定し、
    その範囲内に mask(白=255) が含まれていれば True を返す。

    - mask: 0 or 255 の二値画像 (グレースケール)
    - rect_w, rect_h: 縦長矩形の幅, 高さ
    """
    H, W = mask.shape[:2]

    center_x = W // 2
    center_y = H // 2

    rect_x1 = center_x - rect_w // 2
    rect_y1 = center_y - rect_h // 2
    rect_x2 = rect_x1 + rect_w
    rect_y2 = rect_y1 + rect_h

    roi = mask[rect_y1:rect_y2, rect_x1:rect_x2]

    if np.any(roi == 255):
        return True
    else:
        return False

def draw_center_rect(bg, rect_w, rect_h, color=(255, 0, 0), thickness=2):
    """
    画面中央に (rect_w, rect_h) の矩形を描画して返す
    - bg: 背景画像 (カラー)
    - rect_w, rect_h: 矩形の幅, 高さ
    - color, thickness: 枠線の色と太さ
    """
    H, W = bg.shape[:2]
    center_x = W // 2
    center_y = H // 2

    rect_x1 = center_x - rect_w // 2
    rect_y1 = center_y - rect_h // 2
    rect_x2 = rect_x1 + rect_w
    rect_y2 = rect_y1 + rect_h

    # 矩形を描画
    cv2.rectangle(bg, (rect_x1, rect_y1), (rect_x2, rect_y2), color, thickness)

    return bg


def create_silhouette_alpha(gray_mask, offset_x, offset_y, bg):
    """
    グレースケール値(0～255)を 0.0～1.0 のアルファ値として扱い、
    画素ごとにブレンドして合成する。
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

    alpha_roi = mask_roi.astype(np.float32) / 255.0  # shape=(roi_h, roi_w)

    # ブレンド計算
    # fg_color = (255,255,255)
    # ただし、OpenCV配列なので、バッチで計算するには shape=(roi_h, roi_w, 3) にする
    fg_roi = np.ones_like(bg_roi, dtype=np.float32) * 255.0  # (roi_h, roi_w, 3)

    # alpha_roi は (roi_h, roi_w) なので、(roi_h, roi_w, 1) に変形
    alpha_3d = alpha_roi.reshape(roi_h, roi_w, 1)

    blended_roi = alpha_3d * fg_roi + (1.0 - alpha_3d) * bg_roi.astype(np.float32)

    blended_roi = blended_roi.astype(np.uint8)

    out_img[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = blended_roi

    return out_img

def draw_detection_area_rect(frame, threshold=50, color=(0, 255, 0), thickness=2):
    """
    frame: 可視化したい画像（背景など）
    threshold: is_person_in_center()で使っている threshold
    color, thickness: 矩形の枠色・線の太さ
    """
    h, w = frame.shape[:2]
    
    center_x = w // 2
    center_y = h // 2
    
    # 矩形の左上と右下の座標
    x1 = center_x - threshold
    y1 = center_y - threshold
    x2 = center_x + threshold
    y2 = center_y + threshold

    roi_x, roi_y, roi_w, roi_h = 100, 100, 200, 400
    # roi_x, roi_y, roi_w, roi_h = 780, 140, 400, 800
    
    # 矩形を描画
    # cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.rectangle(frame, (roi_x - roi_w/2, roi_y - roi_h/2), (roi_x + roi_w/2, roi_y + roi_h/2), color, thickness)
    
    return frame

def draw_roi(frame, roi_x, roi_y, roi_w, roi_h, color=(0, 255, 0), thickness=2):
    """
    frame: 描画先の画像
    (roi_x, roi_y, roi_w, roi_h): ROI の左上座標 (roi_x, roi_y) と幅 roi_w, 高さ roi_h
    color: 矩形の色 (B,G,R)
    thickness: 枠線の太さ (負値にすると塗り潰し)
    """
    x1, y1 = roi_x, roi_y
    x2, y2 = roi_x + roi_w, roi_y + roi_h
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    return frame


bg = cv2.imread("background.jpg")
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ROI (検出用の領域) 例
roi_x, roi_y, roi_w, roi_h = 100, 100, 200, 400

STATE_WAITING = 0
STATE_DISPLAY_SILHOUETTE = 1
STATE_PLAY_VIDEO = 2
current_state = STATE_WAITING = 0

center_hold_start_time = 0
start_time = 0
cam_id = 2

model_path = "./models/model.pt"
target_size = (480, 270)
device = "cuda:0"
quantize = False

start_time = time.time()

video_path = "./Ghost4.mp4"

rect_w, rect_h = 600, 960  # 画面中央の縦長矩形

cap_video = cv2.VideoCapture(video_path)
if not cap_video.isOpened():
    raise FileNotFoundError("動画ファイルが見つからないか、読み込みに失敗しました: " + video_path)


video_inference = VideoInference(model_path, target_size, device=device, quantize=quantize, slog3=False)
cap = cv2.VideoCapture(cam_id)
if not cap.isOpened():
    print("Error: Could not open webcam.")
else: 
    while True:
        # (2) 二値画像を取得（既存実装の関数想定）
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # ROI を切り出してマスクを取得
        # roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_h]
        # print(frame.shape, roi_frame.shape)
        # mask = video_inference.inference_frame(roi_frame)
        
        mask = video_inference.inference_frame(frame)

        if current_state == STATE_WAITING:
            # in_center = is_person_in_center(mask, rect_w, rect_h)
            in_center = True

            cv2.imshow("window", bg)

            if in_center:
                if center_hold_start_time == 0:
                    center_hold_start_time = time.time()
                else:
                    if time.time() - center_hold_start_time > 2.0:
                        current_state = STATE_DISPLAY_SILHOUETTE
                        start_time = time.time()
                        center_hold_start_time = 0
            else:
                center_hold_start_time = 0

        elif current_state == STATE_DISPLAY_SILHOUETTE:
            print(current_state)
            elapsed = time.time() - start_time
            if elapsed <= 10:

                bg_h, bg_w = bg.shape[:2]

                scale = 6.0 # 必要に応じて変更
                # mask_h, mask_w = mask.shape[:2]
                # new_w = int(mask_w * scale)
                # new_h = int(mask_h * scale)
                resized_mask = cv2.resize(mask, (bg_w, bg_h), interpolation=cv2.INTER_AREA)

                # offset_x = (bg_w - new_w) // 2
                # offset_y = (bg_h - new_h) // 2

                silhouette_img = create_silhouette_alpha(resized_mask, offset_x=0, offset_y=0, bg=bg)

                alpha = 1
                display_img = cv2.addWeighted(bg, 1-alpha, silhouette_img, alpha, 0)

                cv2.imshow("window", display_img)
            
            else:
                current_state = STATE_PLAY_VIDEO
                print(current_state)

                
        elif current_state == STATE_PLAY_VIDEO:
            ret, video_frame = cap_video.read()
            if not ret:
                # 動画終了 → 0秒に戻してリスタート
                cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_state = STATE_WAITING
                continue

            # 背景サイズに合わせてリサイズ
            video_frame_resized = cv2.resize(video_frame, (bg.shape[1], bg.shape[0]))
            cv2.imshow("window", video_frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()