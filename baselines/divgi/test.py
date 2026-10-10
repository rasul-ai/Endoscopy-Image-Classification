import os
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt
import numpy as np

from models.get_model import get_arch
from utils.reproducibility import set_seeds

# =======================
# Configuration
# =======================
NUM_CLASSES = 15
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.path.join(ROOT, "data", "new_dataset_15")
MODEL_PATH = os.path.join(ROOT, "checkpoints", "divgi.pth")
OUTPUT_DIR = os.path.join(ROOT, "results", "test", "divgi")
DO_MULTIGRANULARITIES = True
N_STEP = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =======================
# Reproducibility
# =======================
set_seeds(42, torch.cuda.is_available())

# =======================
# Load Model
# =======================
print("🔄 Loading model...")
model, mean, std = get_arch(
    model_name="resnet50",
    n_classes=NUM_CLASSES,
    do_multigranularities=DO_MULTIGRANULARITIES,
    n_step=N_STEP,
    do_multilabel=False,
    number_multi_label=[NUM_CLASSES]
)
model = model.to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded successfully.")

# =======================
# Data Loading
# =======================
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_dataset = datasets.ImageFolder(root=f"{DATA_PATH}/test", transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# =======================
# Inference
# =======================
print("🚀 Running inference on test set...")
all_labels, all_preds = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = model(inputs, stage=1)

        if DO_MULTIGRANULARITIES:
            logits_list = outputs[0]
            logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        else:
            logits = outputs[0][-1]

        _, preds = torch.max(logits, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# =======================
# Metrics Calculation
# =======================
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
bacc = balanced_accuracy_score(all_labels, all_preds)

print("\n✅ Inference Metrics:")
print(f"Accuracy          : {acc * 100:.2f}%")
print(f"Precision (macro) : {precision:.4f}")
print(f"Recall (macro)    : {recall:.4f}")
print(f"F1-score (macro)  : {f1:.4f}")
print(f"Balanced Accuracy : {bacc:.4f}")

# =======================
# Classification Report
# =======================
report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4)
print("\n📋 Classification Report:")
print(report)

# =======================
# Confusion Matrix
# =======================
cm = confusion_matrix(all_labels, all_preds, labels=np.arange(NUM_CLASSES))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
plt.figure(figsize=(12, 10))
disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()
print(f"📊 Confusion matrix saved at: {OUTPUT_DIR}/confusion_matrix.png")

# =======================
# Save Metrics
# =======================
metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("===== Test Metrics =====\n")
    f.write(f"Accuracy          : {acc * 100:.2f}%\n")
    f.write(f"Precision (macro) : {precision:.4f}\n")
    f.write(f"Recall (macro)    : {recall:.4f}\n")
    f.write(f"F1-score (macro)  : {f1:.4f}\n")
    f.write(f"Balanced Accuracy : {bacc:.4f}\n\n")
    f.write("===== Classification Report =====\n")
    f.write(report)

print(f"🧾 Metrics and report saved at: {metrics_path}")
