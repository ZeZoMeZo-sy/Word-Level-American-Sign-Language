import pandas as pd
import numpy as np

def get_balanced_df(csv_path, target_count=15):
    df = pd.read_csv(csv_path)
    # Only balance the training set
    train_df = df[df['split'] == 'train'].copy()
    
    balanced_list = []
    
    for label in train_df['label'].unique():
        class_subset = train_df[train_df['label'] == label]
        current_count = len(class_subset)
        
        if current_count >= target_count:
            # If you already have 15+, keep them all
            balanced_list.append(class_subset)
        else:
            # If you have less than 15, duplicate them randomly
            extra_needed = target_count - current_count
            extra_samples = class_subset.sample(n=extra_needed, replace=True)
            balanced_list.append(pd.concat([class_subset, extra_samples]))
            
    return pd.concat(balanced_list).reset_index(drop=True)

# Usage
df_balanced = get_balanced_df(r"C:\Users\ahmad altayar\Desktop\wlasl project\data\row\labels_encoded.csv")
print(df_balanced['label'].value_counts()) # Every class will now show at least 15!

def augment_landmarks(data):
    """
    Safe augmentation for MediaPipe sign language landmarks
    """

    # 1. SMALL gaussian noise (very important, but subtle)
    noise = np.random.normal(0, 0.002, data.shape)
    data = data + noise

    # 2. VERY small random jitter (ONLY slight realism boost)
    jitter = np.random.uniform(-0.005, 0.005)
    data = data + jitter

    return data