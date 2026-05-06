import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# --- SETTINGS ---
dataset_root = r"C:\Users\ahmad altayar\Desktop\wlasl project\data\row"
output_name = "labels_encoded.csv"

def create_labels():
    root = Path(dataset_root)
    data = []

    # 1. Walk through the folders and collect data
    print("Reading folders...")
    for split in ["train", "test", "val"]:
        split_path = root / split
        
        if not split_path.exists():
            print(f"Skipping {split}: Folder not found.")
            continue
            
        for class_folder in sorted(os.listdir(split_path)):
            class_path = split_path / class_folder
            
            if not class_path.is_dir():
                continue
            
            for video_file in os.listdir(class_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = class_path / video_file
                    
                    data.append({
                        'video_path': str(video_path),
                        'label': class_folder,
                        'split': split
                    })

    # 2. Create DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        print("No videos found. Check your dataset path.")
        return

    # 3. Use LabelEncoder to create the numbers
    encoder = LabelEncoder()
    df['label_encoded'] = encoder.fit_transform(df['label'])

    # 4. Save to CSV
    output_path = root / output_name
    df.to_csv(output_path, index=False)

    print("\nSuccess!")
    print(f"File saved to: {output_path}")
    print(f"Total videos: {len(df)}")
    print(f"Unique classes found: {len(encoder.classes_)}")
    
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    create_labels()