import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp

# ====================== CONFIG ======================
CSV_PATH = r"C:\Users\ahmad altayar\Desktop\wlasl project\data\row\labels_encoded.csv"
OUTPUT_ROOT = Path(r"C:\Users\ahmad altayar\Desktop\wlasl project\newmediapipe")

# ====================== INIT MEDIAPIPE ======================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1, 
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ====================== HELPERS ======================
def extract_keypoints(results):
    """
    Extracts 150 features: Pose (66), Left Hand (42), Right Hand (42).
    Uses relative coordinates to improve generalization.
    """
    
    # 1. Pose (33 points * 2: x, y) = 66 values
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten()
        # RELATIVE: Subtract the nose (index 0) from all points to center data
        # This makes the model ignore if you are standing left, right, or center.
        nose_x, nose_y = pose[0], pose[1]
        pose[0::2] -= nose_x
        pose[1::2] -= nose_y
    else:
        pose = np.zeros(33 * 2)
    
    # 2. Left Hand (21 points * 2: x, y) = 42 values
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten()
        # RELATIVE: Subtract the wrist (index 0 of hand) to focus on finger shape
        lh[0::2] -= lh[0]
        lh[1::2] -= lh[1]
    else:
        lh = np.zeros(21 * 2)
    
    # 3. Right Hand (21 points * 2: x, y) = 42 values
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten()
        rh[0::2] -= rh[0]
        rh[1::2] -= rh[1]
    else:
        rh = np.zeros(21 * 2)
    
    return np.concatenate([pose, lh, rh])

def process_video(video_path, save_path):
    cap = cv2.VideoCapture(str(video_path))
    video_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        kp = extract_keypoints(results)
        video_keypoints.append(kp)

    cap.release()

    if video_keypoints:
        np.save(save_path, np.array(video_keypoints))
        return True
    return False

# ====================== MAIN ======================
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting extraction for {len(df)} videos...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Videos"):
        video_path = Path(row['video_path'])
        split = row['split']
        label = row['label']
        video_name = video_path.stem

        save_dir = OUTPUT_ROOT / split / label
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{video_name}.npy"

        if not save_path.exists():
            success = process_video(video_path, save_path)
            
    print(f"\nDone! Data saved to {OUTPUT_ROOT}")