import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.utils.data import DataLoader
from pathlib import Path

# Import your setup
from model import SignLanguageModel
from train import WLASLDataset

# ====================== CONFIG ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\ahmad altayar\Desktop\wlasl project\src\best_sign_model.pth"
CSV_PATH = Path(r"C:\Users\ahmad altayar\Desktop\wlasl project\data\row\labels_encoded.csv")
NPY_ROOT = Path(r"C:\Users\ahmad altayar\Desktop\wlasl project\newmediapipe")

def show_results():
    # 1. Load data and label names
    df = pd.read_csv(CSV_PATH)
    val_df = df[df['split'] == 'val']
    class_names = df[['label_encoded', 'label']].drop_duplicates().sort_values('label_encoded')['label'].tolist()

    # 2. Prepare Data Loader (No augmentation for results)
    val_loader = DataLoader(WLASLDataset(val_df, NPY_ROOT, augment=False), batch_size=32)

    # 3. Load Model (Ensure architecture matches your 300-feature version)
    num_classes = len(class_names)
    model = SignLanguageModel(input_size=300, num_classes=num_classes).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"❌ Error: Your saved model doesn't match the code. {e}")
        return

    all_preds, all_labels = [], []

    # 4. Run Evaluation
    print("🧠 Processing validation data...")
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 5. Calculate F1 Score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\n✅ Final Weighted F1-Score: {f1:.4f}")

    # 6. Classification Report (Precision/Recall)
    print("\n--- PERFORMANCE SUMMARY ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 7. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Confusion Matrix: Predicted vs Actual')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Save and Show
    plt.savefig('final_results_plot.png')
    print("💾 Plot saved as 'final_results_plot.png'")
    plt.show()

if __name__ == "__main__":
    show_results()