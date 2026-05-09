import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from collections import Counter

from data_utils import get_balanced_df, augment_landmarks
from model import SignLanguageModel

# ====================== CONFIG ======================

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NPY_ROOT    = Path(r"C:\Users\ahmad altayar\Desktop\wlasl project\newmediapipe")
CSV_PATH    = Path(r"C:\Users\ahmad altayar\Desktop\wlasl project\data\row\labels_encoded.csv")

TARGET_FRAMES   = 30
BATCH_SIZE      = 32
EPOCHS          = 150
LR              = 2e-4
WEIGHT_DECAY    = 0.03
DROPOUT         = 0.35
TARGET_COUNT    = 80          
LABEL_SMOOTHING = 0.1         # prevents overconfident predictions


# ====================== DATASET ======================

class WLASLDataset(Dataset):
    def __init__(self, df, npy_root, augment=False):
        self.df       = df.reset_index(drop=True)
        self.npy_root = npy_root
        self.augment  = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        video_name = Path(row['video_path']).stem
        npy_path   = self.npy_root / row['split'] / row['label'] / f"{video_name}.npy"

        data = np.load(npy_path).astype(np.float32)

        # ---- FIX 1: pad at START not end ----
        # Zero-padding at the end means the LSTM's last timestep is silence.
        # Padding at the start means the sign always ends at the last frame.
        if data.shape[0] > TARGET_FRAMES:
            idx_seq = np.linspace(0, data.shape[0] - 1, TARGET_FRAMES, dtype=int)
            data = data[idx_seq]
        else:
            pad_len = TARGET_FRAMES - data.shape[0]
            padding = np.zeros((pad_len, data.shape[1]), dtype=np.float32)
            data = np.vstack((padding, data))   # <-- pad at START

        # ---- FIX 2: augment before computing velocity ----
        if self.augment:
            data = augment_landmarks(data)

        # Velocity features
        velocity    = np.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]
        data = np.concatenate([data, velocity], axis=1)

        return (
            torch.tensor(data,                    dtype=torch.float32),
            torch.tensor(int(row['label_encoded']), dtype=torch.long)
        )


# ====================== WEIGHTED SAMPLER ======================

def make_weighted_sampler(df):
    """
    Gives each sample a weight inversely proportional to its class frequency.
    Rare classes get sampled more often — fixes class imbalance without
    discarding any data.
    """
    counts  = Counter(df['label_encoded'].tolist())
    weights = [1.0 / counts[label] for label in df['label_encoded'].tolist()]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ====================== MIXUP ======================

def mixup_batch(x, y, num_classes, alpha=0.3):
    """
    Mixup augmentation: blends two random samples together.
    Forces the model to learn smoother decision boundaries.
    Returns mixed x and soft labels (one-hot floats).
    """
    lam    = np.random.beta(alpha, alpha)
    idx    = torch.randperm(x.size(0)).to(x.device)
    x_mix  = lam * x + (1 - lam) * x[idx]

    # Soft one-hot labels
    y_a    = F.one_hot(y,        num_classes).float()
    y_b    = F.one_hot(y[idx],   num_classes).float()
    y_mix  = lam * y_a + (1 - lam) * y_b

    return x_mix, y_mix


# ====================== TRAINING ======================

def train():
    full_df  = pd.read_csv(CSV_PATH)
    train_df = get_balanced_df(str(CSV_PATH), target_count=TARGET_COUNT)
    val_df   = full_df[full_df['split'] == 'val']

    num_classes = full_df['label_encoded'].nunique()

    print(f"Train samples : {len(train_df)}")
    print(f"Val   samples : {len(val_df)}")
    print(f"Num classes   : {num_classes}")

    train_dataset = WLASLDataset(train_df, NPY_ROOT, augment=True)
    val_dataset   = WLASLDataset(val_df,   NPY_ROOT, augment=False)

    # pin_memory only helps on GPU — avoids warning on CPU
    use_pin_memory = DEVICE.type == 'cuda'

    # Weighted sampler so every class gets fair representation each epoch
    sampler      = make_weighted_sampler(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False,  num_workers=2, pin_memory=use_pin_memory)

    model = SignLanguageModel(
        input_size=300,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(DEVICE)

    # Label smoothing prevents the model from becoming overconfident
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine annealing is smoother than ReduceLROnPlateau for sequence models
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history    = {'t_acc': [], 'v_acc': [], 't_loss': [], 'v_loss': [], 'v_f1': []}
    best_v_acc = 0.0

    print(f"\n🚀 Training on {DEVICE} | {EPOCHS} epochs\n")

    for epoch in range(EPOCHS):
        # -------- TRAIN --------
        model.train()
        t_correct, t_total, t_loss_sum = 0, 0, 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Apply Mixup 50% of the time
            use_mixup = np.random.rand() < 0.5
            if use_mixup:
                x_mix, y_soft = mixup_batch(x, y, num_classes)
                optimizer.zero_grad()
                out  = model(x_mix)
                loss = -(y_soft * F.log_softmax(out, dim=1)).sum(dim=1).mean()
            else:
                optimizer.zero_grad()
                out  = model(x)
                loss = criterion(out, y)

            loss.backward()
            # Gradient clipping — prevents exploding gradients in LSTM
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            t_loss_sum += loss.item()
            t_correct  += (torch.argmax(out, dim=1) == y).sum().item()
            t_total    += y.size(0)

        scheduler.step()

        # -------- VALIDATE --------
        model.eval()
        v_correct, v_total, v_loss_sum = 0, 0, 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y  = x.to(DEVICE), y.to(DEVICE)
                out   = model(x)
                v_loss_sum += criterion(out, y).item()

                preds      = torch.argmax(out, dim=1)
                v_correct  += (preds == y).sum().item()
                v_total    += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # -------- METRICS --------
        avg_t_loss  = t_loss_sum / len(train_loader)
        avg_v_loss  = v_loss_sum / len(val_loader)
        t_acc       = 100 * t_correct / t_total
        v_acc       = 100 * v_correct / v_total
        v_f1        = f1_score(all_labels, all_preds, average='weighted')
        current_lr  = optimizer.param_groups[0]['lr']

        history['t_acc'].append(t_acc)
        history['v_acc'].append(v_acc)
        history['t_loss'].append(avg_t_loss)
        history['v_loss'].append(avg_v_loss)
        history['v_f1'].append(v_f1)

        print(
            f"[{epoch+1:03d}] "
            f"Loss: {avg_t_loss:.4f} | Val Loss: {avg_v_loss:.4f} | "
            f"Acc: {t_acc:.1f}% | Val Acc: {v_acc:.1f}% | "
            f"F1: {v_f1:.4f} | LR: {current_lr:.2e}"
        )

        if v_acc > best_v_acc:
            prev_best  = best_v_acc
            best_v_acc = v_acc
            # Overwrite — only the single best checkpoint is ever kept on disk
            torch.save(model.state_dict(), "best_sign_model_final01.pth")
            if prev_best == 0.0:
                print(f"  ✅ First checkpoint saved — Val Acc: {best_v_acc:.1f}%")
            else:
                print(f"  ✅ Improved {prev_best:.1f}% → {best_v_acc:.1f}% | old checkpoint replaced")

    # -------- PLOT --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(history['t_acc'],  label='Train Acc')
    axes[0].plot(history['v_acc'],  label='Val Acc')
    axes[0].set_title('Accuracy')
    axes[0].legend()

    axes[1].plot(history['t_loss'], label='Train Loss')
    axes[1].plot(history['v_loss'], label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].legend()

    axes[2].plot(history['v_f1'],   label='Val F1', color='green')
    axes[2].set_title('F1 Score')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    print(f"\n Best Val Acc: {best_v_acc:.1f}%")


if __name__ == "__main__":
    train()