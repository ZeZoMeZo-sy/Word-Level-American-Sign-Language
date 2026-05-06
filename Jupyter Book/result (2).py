
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ================= IMPORT YOUR MODEL =================
from model import SignLanguageModel


# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = r"C:\Users\ahmad altayar\Desktop\wlasl project\data\row\labels_encoded.csv"
NPY_ROOT = r"C:\Users\ahmad altayar\Desktop\wlasl project\newmediapipe"
MODEL_PATH = r"models\final_sign_model.pth"

# Put your latest training csv log here
TRAIN_LOG = r"C:\Users\ahmad altayar\Desktop\wlasl project\logs\training_log_20260427_135304.csv"

TARGET_FRAMES = 30
BATCH_SIZE = 32


# ================= HELPERS =================
def adjust_sequence_length(data, target_frames=30):
    current_frames = data.shape[0]

    if current_frames == target_frames:
        return data

    if current_frames > target_frames:
        idx = np.linspace(
            0,
            current_frames - 1,
            target_frames,
            dtype=int
        )
        return data[idx]

    padding = np.zeros(
        (target_frames-current_frames, data.shape[1]),
        dtype=data.dtype
    )

    return np.vstack((data, padding))


# ================= DATASET =================
class WLASLDataset(Dataset):
    def __init__(self, df, npy_root):
        self.df = df.reset_index(drop=True)
        self.npy_root = Path(npy_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_name = Path(row['video_path']).stem

        npy_path = (
            self.npy_root /
            row['split'] /
            row['label'] /
            f"{video_name}.npy"
        )

        data = np.load(npy_path).astype(np.float32)
        data = adjust_sequence_length(data)

        label = int(row['label_encoded'])

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


# ================= EVALUATE =================
def evaluate_model():

    print("Loading test data...")

    df = pd.read_csv(CSV_PATH)
    test_df = df[df['split']=='test'].reset_index(drop=True)

    num_classes = df['label_encoded'].nunique()

    test_dataset = WLASLDataset(test_df, NPY_ROOT)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Loading model...")

    model = SignLanguageModel(
        num_classes=num_classes
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    all_preds=[]
    all_labels=[]

    correct=0
    total=0

    with torch.no_grad():
        for data,labels in test_loader:
            data=data.to(DEVICE)
            labels=labels.to(DEVICE)

            outputs=model(data)
            preds=torch.argmax(outputs,dim=1)

            total+=labels.size(0)
            correct+=(preds==labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy=100*correct/total

    print("\n"+"="*50)
    print(f"TEST ACCURACY: {accuracy:.2f}%")
    print("="*50)


    # ===== Classification report =====
    print("\nClassification Report:\n")
    print(
        classification_report(
            all_labels,
            all_preds,
            zero_division=0
        )
    )


    # ===== Confusion matrix =====
    print("Showing confusion matrix...")

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    plt.figure(figsize=(12,12))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(
        xticks_rotation=90,
        values_format='d'
    )
    plt.title("Confusion Matrix")
    plt.show()


# ================= TRAINING CURVES =================
def plot_training_logs():

    try:
        log_df = pd.read_csv(TRAIN_LOG)
    except Exception:
        print(
            "Change TRAIN_LOG path to your real log file first."
        )
        return


    epochs = log_df['epoch']


    # Loss plot
    plt.figure(figsize=(10,6))
    plt.plot(
        epochs,
        log_df['train_loss'],
        label='Train Loss'
    )
    plt.plot(
        epochs,
        log_df['val_loss'],
        label='Val Loss'
    )

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.show()


    # Accuracy plot
    plt.figure(figsize=(10,6))
    plt.plot(
        epochs,
        log_df['train_acc'],
        label='Train Accuracy'
    )
    plt.plot(
        epochs,
        log_df['val_acc'],
        label='Val Accuracy'
    )

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.show()


    # Learning rate plot
    plt.figure(figsize=(10,6))
    plt.plot(
        epochs,
        log_df['lr']
    )

    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.show()


# ================= MAIN =================
if __name__ == "__main__":

    print("1) Plotting training curves...")
    plot_training_logs()

    print("\n2) Running test evaluation...")
    evaluate_model()




