import os
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# =======================
# Setup
# =======================
BATCH_SIZE = 16
NUM_CLASSES = 15
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = os.path.join(ROOT, "results", "test", "resnet50")
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(ROOT, "checkpoints", "resnet50.pth")

# =======================
# Class Names (from train dataset)
# =======================
train_dataset = datasets.ImageFolder(root=os.path.join(ROOT, "data", "new_dataset_15", "train"))
class_names = train_dataset.classes

# =======================
# Transform for Test Dataset
# =======================
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# =======================
# Load Test Dataset
# =======================
test_dataset = datasets.ImageFolder(root=os.path.join(ROOT, "data", "new_dataset_15", "test"), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =======================
# Load Model
# =======================
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.to(DEVICE)
model.eval()

# =======================
# Inference
# =======================
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# =======================
# Metrics
# =======================
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
bacc = balanced_accuracy_score(y_true, y_pred)

print(f"✅ Inference Results:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Balanced Accuracy: {bacc:.4f}")

# =======================
# Confusion Matrix
# =======================
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix - ResNet50")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "resnet50_confusion_matrix.png"))
plt.close()

# =======================
# Save Metrics to File
# =======================
metrics_path = os.path.join(output_dir, "resnet50_test_metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write(f"Balanced Accuracy: {bacc:.4f}\n")

print(f"📊 Confusion matrix and metrics saved in '{output_dir}'")
