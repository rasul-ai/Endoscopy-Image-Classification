import os
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt

# =======================
# Settings
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 15
IMAGE_SIZE = 224
MODEL_PATH = os.path.join(ROOT, "checkpoints", "balanced_mixup.pth")
OUTPUT_DIR = os.path.join(ROOT, "results", "test", "balanced_mixup")
os.makedirs(OUTPUT_DIR, exist_ok=True)
VAL_DIR = os.path.join(ROOT, "data", "new_dataset_15", "test")  # test split

# Data Transform
# =======================
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# =======================
# Load Dataset
# =======================
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = val_dataset.classes

# =======================
# Load Model
# =======================
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =======================
# Inference
# =======================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# =======================
# Compute Metrics
# =======================
precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
bacc = balanced_accuracy_score(all_labels, all_preds)
accuracy = 100 * np.mean(all_preds == all_labels)

print(f"Validation Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Balanced Acc: {bacc:.4f}")

# =======================
# Confusion Matrix
# =======================
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(12, 12))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "test_confusion_matrix.png"))
plt.show()
