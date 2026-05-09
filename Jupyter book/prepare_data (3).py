import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# --- SETTINGS ---
dataset_root = r"C:\Users\ahmad altayar\Desktop\wlasl project\data\row"
output_name = "labels_encoded.csv"
# Filtering out the class with no real data 
MIN_SAMPLES_PER_CLASS = 5  

def create_labels():
    root = Path(dataset_root)
    data = []

    # 1. Walk through the folders and collect data
    print("start to read the folders")
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
        print("No videos found check the pathes")
        return

    # 3. CLEANING: Filter out classes with insufficient data to generalize
    # Count how many videos each class has across the whole dataset
    class_counts = df['label'].value_counts()
    valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
    
    initial_count = len(df)
    df = df[df['label'].isin(valid_classes)].reset_index(drop=True)
    
    print(f"Filtered {initial_count - len(df)} videos from classes with < {MIN_SAMPLES_PER_CLASS} samples.")

    # 4. Use LabelEncoder to create the numbers
    encoder = LabelEncoder()
    df['label_encoded'] = encoder.fit_transform(df['label'])

    # 5. Save to CSV
    output_path = root / output_name
    df.to_csv(output_path, index=False)

    print("\nSuccess!")
    print(f"File saved to: {output_path}")
    print(f"Total videos remaining: {len(df)}")
    print(f"Unique classes kept: {len(encoder.classes_)}")
    
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    create_labels()