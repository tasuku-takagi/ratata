import cv2
import numpy as np
import time

import sys
sys.path.append("../")
from ratata.inference.video_inference import VideoInference

# (1) 初期準備
bg = cv2.imread("background.jpg")  # 背景画像
# フルスクリーンウィンドウ作成
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

STATE_WAITING = 0
STATE_DISPLAY_SILHOUETTE = 1
STATE_FADE_OUT = 2
STATE_PLAY_VIDEO = 3

current_state = STATE_WAITING
start_time = 0
center_hold_start_time = 0

cam_id = 0

model_path = "./models/model.pt"
target_size = (320, 180)
device = "cpu"
quantize = False

video_inference = VideoInference(model_path, target_size, device=device, quantize=quantize, slog3=slog3)
cap = cv2.VideoCapture(cam_id)  # デフォルトカメラ(0)を使用
if not cap.isOpened():
    print("Error: Could not open webcam.")
    return

while True:
    # (2) 二値画像を取得（既存実装の関数想定）
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # 推論の実行
    start_time = time.time()
    mask = video_inference.inference_frame(frame)
    # mask = get_binary_mask_from_AI()  
    
    # (3) 現在状態ごとに処理分岐
    if current_state == STATE_WAITING:
        # 背景画像表示
        cv2.imshow("window", bg)
        
        # 人が中央にいるか判定
        if is_center(mask):
            # はじめて中央に入った瞬間なら時刻を記録
            if center_hold_start_time == 0:
                center_hold_start_time = time.time()
            else:
                # 一定秒数経過したら状態遷移
                if time.time() - center_hold_start_time > 2.0:  # 2秒以上留まった
                    current_state = STATE_DISPLAY_SILHOUETTE
                    start_time = time.time()
                    center_hold_start_time = 0
        else:
            center_hold_start_time = 0

    elif current_state == STATE_DISPLAY_SILHOUETTE:
        elapsed = time.time() - start_time
        if elapsed <= 10.0:
            # 10秒間はシルエット表示
            alpha = 1.0  # シルエットの濃さを固定 or フェードインにしたいなら timeベースで変化
            silhouette_img = create_silhouette(mask, bg.shape)  # 真っ白にしたり、何かしらの描画
            display_img = cv2.addWeighted(bg, 1-alpha, silhouette_img, alpha, 0)
            cv2.imshow("window", display_img)
        else:
            # 10秒超えたらフェードアウト状態へ
            current_state = STATE_FADE_OUT
            start_time = time.time()

    elif current_state == STATE_FADE_OUT:
        elapsed = time.time() - start_time
        if elapsed <= 2.0:  # 2秒で消えていく
            alpha = elapsed / 2.0  # 0→1に変化
            silhouette_img = create_silhouette(mask, bg.shape)
            # alphaが大きいほどシルエットが消えていくなら演出に応じて調整
            display_img = cv2.addWeighted(bg, 1, silhouette_img, (1 - alpha), 0)
            cv2.imshow("window", display_img)
        else:
            # 2秒経過したら次の動画再生へ
            current_state = STATE_PLAY_VIDEO

    elif current_state == STATE_PLAY_VIDEO:
        # 動画再生関数など作って、そこで完了するまでブロック or コールバック
        play_video_fullscreen("another_video.mp4")  # 終わったら戻る
        current_state = STATE_WAITING

    # キー入力チェック
    if cv2.waitKey(1) & 0xFF == 27:  # ESCキー
        break

cv2.destroyAllWindows()
