import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime

from data_utils import get_balanced_df, augment_landmarks
from model import SignLanguageModel


# ====================== HELPER ======================
def adjust_sequence_length(data, target_frames=24):
    current_frames = data.shape[0]

    if current_frames == target_frames:
        return data

    if current_frames > target_frames:
        indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
        return data[indices]

    padding = np.zeros((target_frames - current_frames, data.shape[1]), dtype=data.dtype)
    return np.vstack((data, padding))


# ====================== DATASET ======================
class WLASLDataset(Dataset):
    def __init__(self, df, npy_root, target_frames=24, augment=False):
        self.df = df.reset_index(drop=True)
        self.npy_root = Path(npy_root)
        self.target_frames = target_frames
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = Path(row['video_path']).stem
        npy_path = self.npy_root / row['split'] / row['label'] / f"{video_name}.npy"

        data = np.load(npy_path).astype(np.float32)
        data = adjust_sequence_length(data, self.target_frames)

        # ================= NORMALIZATION =================
        pose = data[:, :132]
        anchor_x = pose[:, 0]
        anchor_y = pose[:, 1]

        data[:, 0::2] -= anchor_x[:, None]
        data[:, 1::2] -= anchor_y[:, None]

        # ================= AUGMENTATION =================
        if self.augment:
            data = augment_landmarks(data)

        # ================= VELOCITY =================
        velocity = np.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]

        data = np.concatenate([data, velocity], axis=1)

        label = int(row['label_encoded'])

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ====================== VALIDATION ======================
def validate(model, loader, criterion, device):
    model.eval()
    loss_total, correct, total = 0, 0, 0

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            loss_total += loss.item()
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return loss_total / len(loader), 100 * correct / total


# ====================== TRAIN ======================
def train():
    EPOCHS = 100
    BATCH_SIZE = 24
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NPY_ROOT = r"C:\Users\ahmad altayar\Desktop\wlasl project\newmediapipe"
    CSV_PATH = r"C:\Users\ahmad altayar\Desktop\wlasl project\data\row\labels_encoded.csv"

    # ================= LOGGING =================

    LOG_DIR = Path(r"C:\Users\ahmad altayar\Desktop\wlasl project\logs")
    MODEL_DIR = Path(r"C:\Users\ahmad altayar\Desktop\wlasl project\models")
    LOG_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_csv = LOG_DIR / f"log_{timestamp}.csv"

    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])

    print("🚀 Device:", DEVICE)

    # ================= DATA =================
    full_df = pd.read_csv(CSV_PATH)

    train_df = get_balanced_df(CSV_PATH, target_count=15)
    train_df = train_df[train_df['split'] == 'train']

    val_df = full_df[full_df['split'] == 'val']

    print("Train:", len(train_df), "Val:", len(val_df))

    train_loader = DataLoader(
        WLASLDataset(train_df, NPY_ROOT, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        WLASLDataset(val_df, NPY_ROOT, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # ================= MODEL =================
    num_classes = full_df['label_encoded'].nunique()

    model = SignLanguageModel(
        input_size=516,   # IMPORTANT (258 + velocity)
        num_classes=num_classes
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # ================= EARLY STOP =================
    best_val = float("inf")
    patience = 20
    counter = 0
    best_path = MODEL_DIR / "best_model.pth"

    # ================= TRAIN LOOP =================
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for data, labels in pbar:
            data, labels = data.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Val {val_acc:.2f}%")

        # LOG
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, train_loss, val_loss, train_acc, val_acc, lr])

        scheduler.step(val_loss)

        # ================= EARLY STOP =================
        if val_loss < best_val:
            best_val = val_loss
            counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early stopping")
                break

    print("Best model saved at:", best_path)


if __name__ == "__main__":
    train()