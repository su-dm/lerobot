import time

import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # try with & without
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
print(
    "negotiated:",
    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    cap.get(cv2.CAP_PROP_FPS),
    cap.get(cv2.CAP_PROP_FOURCC),
)
t = []
for _ in range(120):
    s = time.perf_counter()
    ok, frame = cap.read()
    t.append(time.perf_counter() - s)
import statistics as st

print(f"read mean={st.mean(t) * 1e3:.1f}ms  => max sustainable {1 / st.mean(t):.1f} fps")
s = time.perf_counter()
cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print("cvtColor:", (time.perf_counter() - s) * 1e3, "ms")
