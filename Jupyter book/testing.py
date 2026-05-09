import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)
from torch.utils.data import Dataset, DataLoader

from model import SignLanguageModel

# ====================== CONFIG ======================

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\ahmad altayar\Desktop\wlasl project\best_sign_model.pth"
CSV_PATH   = Path(r"C:\Users\ahmad altayar\Desktop\wlasl project\data\row\labels_encoded.csv")
NPY_ROOT   = Path(r"C:\Users\ahmad altayar\Desktop\wlasl project\newmediapipe")

TARGET_FRAMES = 30
BATCH_SIZE    = 32

# ====================== DATASET ======================

class WLASLTestDataset(Dataset):
    def __init__(self, df, npy_root):
        self.df       = df.reset_index(drop=True)
        self.npy_root = npy_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        video_name = Path(row['video_path']).stem
        npy_path   = self.npy_root / row['split'] / row['label'] / f"{video_name}.npy"

        data = np.load(npy_path).astype(np.float32)

        # Same preprocessing as training — pad at START
        if data.shape[0] > TARGET_FRAMES:
            idx_seq = np.linspace(0, data.shape[0] - 1, TARGET_FRAMES, dtype=int)
            data    = data[idx_seq]
        else:
            pad_len = TARGET_FRAMES - data.shape[0]
            padding = np.zeros((pad_len, data.shape[1]), dtype=np.float32)
            data    = np.vstack((padding, data))

        # Velocity features — same as training
        velocity     = np.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]
        data         = np.concatenate([data, velocity], axis=1)

        return (
            torch.tensor(data,                      dtype=torch.float32),
            torch.tensor(int(row['label_encoded']), dtype=torch.long)
        )


# ====================== EVALUATION ======================

def evaluate():
    full_df  = pd.read_csv(CSV_PATH)
    test_df  = full_df[full_df['split'] == 'test'].reset_index(drop=True)

    num_classes = full_df['label_encoded'].nunique()
    label_map   = (
        full_df[['label_encoded', 'label']]
        .drop_duplicates()
        .sort_values('label_encoded')
    )
    class_names = label_map['label'].tolist()

    print(f"Test samples : {len(test_df)}")
    print(f"Num classes  : {num_classes}")

    test_loader = DataLoader(
        WLASLTestDataset(test_df, NPY_ROOT),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Load model
    model = SignLanguageModel(
        input_size=300,
        num_classes=num_classes
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"\nModel loaded from: {MODEL_PATH}")

    # -------- INFERENCE --------
    all_preds     = []
    all_labels    = []
    all_probs     = []   # needed for top-k accuracy

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y  = x.to(DEVICE), y.to(DEVICE)
            out   = model(x)
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ====================== METRICS ======================

    all_class_indices = list(range(num_classes))  # all 100 classes, even if some missing from test

    top1_acc = 100 * (all_preds == all_labels).mean()
    top3_acc = 100 * top_k_accuracy_score(all_labels, all_probs, k=3, labels=all_class_indices)
    top5_acc = 100 * top_k_accuracy_score(all_labels, all_probs, k=5, labels=all_class_indices)
    f1_w     = f1_score(all_labels, all_preds, average='weighted')
    f1_mac   = f1_score(all_labels, all_preds, average='macro')

    print("\n" + "="*55)
    print("              TEST RESULTS")
    print("="*55)
    print(f"  Top-1 Accuracy  : {top1_acc:.2f}%")
    print(f"  Top-3 Accuracy  : {top3_acc:.2f}%")
    print(f"  Top-5 Accuracy  : {top5_acc:.2f}%")
    print(f"  F1 (weighted)   : {f1_w:.4f}")
    print(f"  F1 (macro)      : {f1_mac:.4f}")
    print("="*55)

    # ====================== PER-CLASS ACCURACY ======================

    print("\n--- Per-Class Accuracy ---\n")
    per_class_acc = []
    for cls_idx, cls_name in enumerate(class_names):
        mask    = all_labels == cls_idx
        if mask.sum() == 0:
            continue
        acc     = 100 * (all_preds[mask] == all_labels[mask]).mean()
        count   = mask.sum()
        per_class_acc.append({'class': cls_name, 'accuracy': acc, 'samples': count})

    per_class_df = pd.DataFrame(per_class_acc).sort_values('accuracy', ascending=False)

    # Print top 10 best and worst
    print("Top 10 Best Recognized Signs:")
    print(per_class_df.head(10).to_string(index=False))
    print("\nTop 10 Worst Recognized Signs:")
    print(per_class_df.tail(10).to_string(index=False))

    # Save full per-class report
    per_class_df.to_csv("per_class_accuracy.csv", index=False)
    print("\nFull per-class report saved → per_class_accuracy.csv")

    # ====================== PLOTS ======================

    fig = plt.figure(figsize=(20, 16))

    # ---- 1. Per-class accuracy bar chart ----
    ax1 = fig.add_subplot(2, 2, (1, 2))
    colors = ['#2ecc71' if a >= 60 else '#e67e22' if a >= 40 else '#e74c3c'
              for a in per_class_df['accuracy']]
    bars = ax1.barh(per_class_df['class'], per_class_df['accuracy'], color=colors)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Per-Class Test Accuracy\n(green ≥60% | orange ≥40% | red <40%)')
    ax1.axvline(x=top1_acc, color='navy', linestyle='--', linewidth=1.5,
                label=f'Overall: {top1_acc:.1f}%')
    ax1.legend()
    ax1.invert_yaxis()

    # ---- 2. Top-K accuracy bar chart ----
    ax2 = fig.add_subplot(2, 2, 3)
    topk_vals   = [top1_acc, top3_acc, top5_acc]
    topk_labels = ['Top-1', 'Top-3', 'Top-5']
    bar_colors  = ['#3498db', '#2ecc71', '#9b59b6']
    bars2 = ax2.bar(topk_labels, topk_vals, color=bar_colors, width=0.5)
    for bar, val in zip(bars2, topk_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Top-K Accuracy')

    # ---- 3. Confidence histogram ----
    ax3 = fig.add_subplot(2, 2, 4)
    correct_mask   = all_preds == all_labels
    top1_confs     = all_probs[np.arange(len(all_labels)), all_labels]
    ax3.hist(top1_confs[correct_mask],  bins=30, alpha=0.6, color='green', label='Correct')
    ax3.hist(top1_confs[~correct_mask], bins=30, alpha=0.6, color='red',   label='Wrong')
    ax3.set_xlabel('Model Confidence for True Class')
    ax3.set_ylabel('Count')
    ax3.set_title('Confidence Distribution\n(Correct vs Wrong Predictions)')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plots saved → test_results.png")

    # ====================== CONFUSION MATRIX ======================
    # Only plot if ≤ 50 classes — otherwise it becomes unreadable

    if num_classes <= 50:
        cm = confusion_matrix(all_labels, all_preds)
        fig2, ax = plt.subplots(figsize=(18, 16))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=0.3
        )
        ax.set_xlabel('Predicted', fontsize=13)
        ax.set_ylabel('True',      fontsize=13)
        ax.set_title('Confusion Matrix', fontsize=15)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0,  fontsize=8)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Confusion matrix saved → confusion_matrix.png")
    else:
        # For large class sets: save top-20 most confused pairs instead
        cm        = confusion_matrix(all_labels, all_preds)
        np.fill_diagonal(cm, 0)   # remove correct predictions
        confused  = []
        for i in range(num_classes):
            for j in range(num_classes):
                if cm[i, j] > 0:
                    confused.append({
                        'true':       class_names[i],
                        'predicted':  class_names[j],
                        'count':      cm[i, j]
                    })
        confused_df = pd.DataFrame(confused).sort_values('count', ascending=False)
        print("\nTop 20 Most Confused Sign Pairs:")
        print(confused_df.head(20).to_string(index=False))
        confused_df.to_csv("confused_pairs.csv", index=False)
        print("Full confusion pairs saved → confused_pairs.csv")

    # ====================== SKLEARN REPORT ======================
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    print("\nDetailed classification report saved → classification_report.txt")


if __name__ == "__main__":
    evaluate()