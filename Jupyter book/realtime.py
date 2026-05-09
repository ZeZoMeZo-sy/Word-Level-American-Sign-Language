import cv2
import torch
import numpy as np
import pandas as pd
from collections import deque, Counter
import mediapipe as mp
from pathlib import Path
import time

from model import SignLanguageModel

# ====================== CONFIG ======================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"C:\Users\ahmad altayar\Desktop\wlasl project\src\best_sign_model_final01.pth"
CSV_PATH = r"C:\Users\ahmad altayar\Desktop\wlasl project\data\row\labels_encoded.csv"

TARGET_FRAMES = 30
PRED_INTERVAL = 5
CONF_THRESHOLD = 60

# ====================== LABELS ======================

df = pd.read_csv(CSV_PATH)

label_map = (
    df[['label_encoded', 'label']]
    .drop_duplicates()
    .sort_values('label_encoded')
)

class_names = label_map['label'].tolist()

# ====================== MODEL ======================

model = SignLanguageModel(
    input_size=300,
    num_classes=len(class_names)
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model Loaded ✔")

# ====================== MEDIAPIPE ======================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Custom drawing specs for cleaner visuals
pose_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3)
hand_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)

# ====================== KEYPOINTS ======================

def extract_keypoints(results):

    # -------- POSE --------
    if results.pose_landmarks:
        pose = np.array([[p.x, p.y] for p in results.pose_landmarks.landmark]).flatten()
        # Normalize relative to nose (landmark 0)
        nose_x, nose_y = pose[0], pose[1]
        pose[0::2] -= nose_x
        pose[1::2] -= nose_y
    else:
        pose = np.zeros(33 * 2)

    # -------- LEFT HAND --------
    if results.left_hand_landmarks:
        lh = np.array([[p.x, p.y] for p in results.left_hand_landmarks.landmark]).flatten()
        # Normalize relative to wrist (landmark 0)
        lh[0::2] -= lh[0]
        lh[1::2] -= lh[1]
    else:
        lh = np.zeros(21 * 2)

    # -------- RIGHT HAND --------
    if results.right_hand_landmarks:
        rh = np.array([[p.x, p.y] for p in results.right_hand_landmarks.landmark]).flatten()
        # Normalize relative to wrist (landmark 0)
        rh[0::2] -= rh[0]
        rh[1::2] -= rh[1]
    else:
        rh = np.zeros(21 * 2)

    return np.concatenate([pose, lh, rh])

# ====================== REALTIME ======================

def run_realtime():

    cap = cv2.VideoCapture(0)

    sequence = deque(maxlen=TARGET_FRAMES)
    pred_history = deque(maxlen=10)

    stable_pred = "Waiting..."
    stable_conf = 0.0

    frame_count = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)

            # Convert back to BGR for drawing
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # ================= DRAW LANDMARKS ON IMAGE =================

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=pose_drawing_spec,
                    connection_drawing_spec=connection_drawing_spec
                )

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_drawing_spec,
                    connection_drawing_spec=connection_drawing_spec
                )

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_drawing_spec,
                    connection_drawing_spec=connection_drawing_spec
                )

            # ================= KEYPOINTS =================

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            frame_count += 1

            # ================= PREDICTION =================

            if len(sequence) == TARGET_FRAMES and frame_count % PRED_INTERVAL == 0:

                data = np.array(sequence, dtype=np.float32)

                # Velocity features (same as training)
                velocity = np.zeros_like(data)
                velocity[1:] = data[1:] - data[:-1]

                final_input = np.concatenate([data, velocity], axis=1)

                tensor = torch.tensor(final_input).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out = model(tensor)
                    probs = torch.softmax(out, dim=1)
                    conf, pred = torch.max(probs, dim=1)

                    conf = conf.item() * 100
                    label = class_names[pred.item()]

                # ================= STABILITY =================

                if conf > CONF_THRESHOLD:
                    pred_history.append(label)
                    stable_pred = Counter(pred_history).most_common(1)[0][0]
                    stable_conf = conf
                else:
                    stable_pred = "No Sign"
                    stable_conf = conf

            # ================= UI (drawn on image) =================

            # Top bar background
            cv2.rectangle(image, (0, 0), (image.shape[1], 70), (0, 0, 0), -1)

            # Sign label
            cv2.putText(
                image,
                f"SIGN: {stable_pred}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Confidence score
            cv2.putText(
                image,
                f"CONF: {stable_conf:.1f}%",
                (600, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Buffer fill indicator (how many frames collected out of TARGET_FRAMES)
            buf_filled = int((len(sequence) / TARGET_FRAMES) * 200)
            cv2.rectangle(image, (20, image.shape[0] - 30), (220, image.shape[0] - 10), (50, 50, 50), -1)
            cv2.rectangle(image, (20, image.shape[0] - 30), (20 + buf_filled, image.shape[0] - 10), (0, 200, 255), -1)
            cv2.putText(
                image,
                "Buffer",
                (225, image.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

            # Show the final image (with landmarks + UI)
            cv2.imshow("WLASL Real-Time", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()