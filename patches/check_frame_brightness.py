"""v_004의 frame brightness 측정 (title card 시작 frame 찾기)."""
import cv2
import numpy as np

cap = cv2.VideoCapture('/workspace/media/output/test3_lipsync/chunks/v_004.mp4')
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"v_004.mp4: {n} frames")
print(f"frame_idx | mean_brightness | std_brightness")
print("-" * 50)

for i in range(n):
    ret, f = cap.read()
    if not ret:
        break
    if i % 10 == 0 or i > 200:  # title card 후보 영역은 매 frame
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        mean_b = float(gray.mean())
        std_b = float(gray.std())
        if i > 200 or i % 50 == 0:
            print(f"  {i:4d}    |    {mean_b:6.1f}     |    {std_b:6.1f}")
cap.release()
