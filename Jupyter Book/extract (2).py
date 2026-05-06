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
# Using Holistic is faster than calling Hands and Pose separately
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1, # 0=fast, 1=balanced, 2=heavy
    min_detection_confidence=0.3,# Lower this so it tries harder to find hands
    min_tracking_confidence=0.3# Lower this so it doesn't give up as easily
)

# ====================== HELPERS ======================
def extract_keypoints(results):
    # Pose (33 points * 4 values: x, y, z, visibility)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Left Hand (21 points * 3 values: x, y, z)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right Hand (21 points * 3 values: x, y, z)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

def process_video(video_path, save_path):
    cap = cv2.VideoCapture(str(video_path))
    video_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        # Extract and append
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
        
        # We use the 'split' and 'label' columns directly from your CSV!
        # This guarantees Train, Test, and Validation are all included.
        split = row['split']
        label = row['label']
        video_name = video_path.stem

        save_dir = OUTPUT_ROOT / split / label
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{video_name}.npy"

        # Only process if file doesn't exist (helpful if script crashes)
        if not save_path.exists():
            success = process_video(video_path, save_path)
            
    print(f"\n Done! Data saved to {OUTPUT_ROOT}")