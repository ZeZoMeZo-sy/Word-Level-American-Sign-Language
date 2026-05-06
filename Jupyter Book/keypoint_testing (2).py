import numpy as np
import cv2

# Load your file
data = np.load(r"C:\Users\ahmad altayar\Desktop\wlasl project\mediapipe\test\accident\00634.npy") 

canvas = np.zeros((600, 600, 3), dtype=np.uint8)

for frame in data:
    canvas.fill(0)
    
    # 1. Draw POSE (Indices 0 to 131) - GREEN
    for i in range(0, 132, 4):
        x, y = int(frame[i] * 600), int(frame[i+1] * 600)
        if x > 0 and y > 0: cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)

    # 2. Draw LEFT HAND (Indices 132 to 194) - RED
    lh_start = 132
    for i in range(lh_start, lh_start + 63, 3):
        x, y = int(frame[i] * 600), int(frame[i+1] * 600)
        if x > 0 and y > 0: cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)

    # 3. Draw RIGHT HAND (Indices 195 to 257) - BLUE
    rh_start = 195
    for i in range(rh_start, rh_start + 63, 3):
        x, y = int(frame[i] * 600), int(frame[i+1] * 600)
        if x > 0 and y > 0: cv2.circle(canvas, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Check Landmarks", canvas)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()