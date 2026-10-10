"""
Evaluate a trained CGLE / ablation checkpoint on the test split (paper Tables 1 and 6).

    python scripts/test.py --ckpt checkpoints/cgle.pth
    python scripts/test.py --model fusion_no_ca --ckpt checkpoints/ablation_fusion_no_ca.pth --out results/test/fusion_no_ca
"""
import os
import sys
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cgle import MODELS
from cgle.data import preprocess_dataset

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", choices=sorted(MODELS), default="cgle")
parser.add_argument("--ckpt", default="checkpoints/cgle.pth")
parser.add_argument("--data", default="data/new_dataset_15")
parser.add_argument("--out", default="results/test/cgle")
args = parser.parse_args()

# ------------------- Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load checkpoint first to extract class ordering
model_path = args.ckpt
assert os.path.exists(model_path), f"No checkpoint found at {model_path}"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

# Restore the class order that was used during training
_c2i = checkpoint.get("class_to_idx", None)
if _c2i is not None:
    class_order = [k for k, v in sorted(_c2i.items(), key=lambda x: x[1])]
    print(f"Class order from checkpoint: {class_order}")
else:
    class_order = None
    print("WARNING: checkpoint has no class_to_idx — using sorted alphabetical order.")

# Load model
model = MODELS[args.model]().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded epoch {checkpoint.get('epoch','?')}  "
      f"val_F1={checkpoint.get('val_f1', float('nan')):.4f}")

# ------------------- Load Test Data -------------------
data = preprocess_dataset(root_dir=args.data, class_order=class_order)
test_loader = data.test_dataloader
idx_to_class = {v: k for k, v in data.test_dataset.class_to_idx.items()}
class_names  = [idx_to_class[i] for i in range(len(idx_to_class))]
print(f"Classes: {class_names}")

# ------------------- Inference -------------------
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

# ------------------- Metrics -------------------
acc       = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print("\nTest Set Evaluation:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

os.makedirs(args.out, exist_ok=True)

# ------------------- Confusion Matrix -------------------
cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(args.out, "test_confusion_matrix.png"))
plt.close()

pd.DataFrame([{
    'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1 Score': f1
}]).to_csv(os.path.join(args.out, "test_metrics.csv"), index=False)
print(f"Saved: {args.out}/test_confusion_matrix.png, {args.out}/test_metrics.csv")
