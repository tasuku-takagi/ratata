import cv2

for index in range(10):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"camid: {index} is open")
    else:
        pass
