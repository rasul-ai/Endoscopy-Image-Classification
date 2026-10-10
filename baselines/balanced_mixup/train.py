import os
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

# =======================
# Hyperparameters
# =======================
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_CLASSES = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 224
alpha = 0.1   # for Balanced-MixUp
# =======================
# Output Directory
# =======================
output_dir = os.path.join(ROOT, "runs", "balanced_mixup")
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "resnet50_model_checkpoint.pth")
csv_path = os.path.join(output_dir, "training_log.csv")

# =======================
# Data Transforms
# =======================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# =======================
# Load Dataset
# =======================
train_dataset = datasets.ImageFolder(root=os.path.join(ROOT, "data", "new_dataset_15", "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(ROOT, "data", "new_dataset_15", "val"), transform=val_transform)
class_names = train_dataset.classes

# Weighted Sampler for class balance
class_counts = np.bincount(train_dataset.targets)
class_weights = 1. / class_counts
sample_weights = [class_weights[t] for t in train_dataset.targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =======================
# Balanced-MixUp Function
# =======================
def balanced_mixup(x, y, alpha=0.1):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_onehot = torch.nn.functional.one_hot(y, NUM_CLASSES).float()
    y_shuffled = torch.nn.functional.one_hot(y[index], NUM_CLASSES).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_shuffled
    return mixed_x, mixed_y

# =======================
# Load Pretrained ResNet-50
# =======================
resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(resnet50.fc.in_features, NUM_CLASSES)
resnet50.to(DEVICE)

# =======================
# Loss, Optimizer, Scheduler
# =======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6)

# =======================
# Training Loop
# =======================
train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []
train_f1s, val_f1s = [], []
train_baccs, val_baccs = [], []

best_val_loss = float("inf")
early_stop_counter = 0
patience = 10

for epoch in range(NUM_EPOCHS):
    resnet50.train()
    total_loss, all_labels, all_preds = 0, [], []

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Apply Balanced-MixUp to 50% batches
        if np.random.rand() < 0.5:
            images, labels_mix = balanced_mixup(images, labels, alpha)
            outputs = resnet50(images)
            loss = -(labels_mix * torch.log_softmax(outputs, dim=1)).sum(dim=1).mean()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        else:
            outputs = resnet50(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Train metrics
    avg_train_loss = total_loss / len(train_loader)
    train_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    train_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    train_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    train_bacc = balanced_accuracy_score(all_labels, all_preds)

    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)
    train_precisions.append(train_prec)
    train_recalls.append(train_rec)
    train_f1s.append(train_f1)
    train_baccs.append(train_bacc)

    # Validation
    resnet50.eval()
    val_loss, val_preds, val_labels = 0, [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = resnet50(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * np.mean(np.array(val_preds) == np.array(val_labels))
    val_prec = precision_score(val_labels, val_preds, average="macro", zero_division=0)
    val_rec = recall_score(val_labels, val_preds, average="macro", zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
    val_bacc = balanced_accuracy_score(val_labels, val_preds)

    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)
    val_precisions.append(val_prec)
    val_recalls.append(val_rec)
    val_f1s.append(val_f1)
    val_baccs.append(val_bacc)

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}, Prec: {train_prec:.4f}, "
          f"Rec: {train_rec:.4f}, F1: {train_f1:.4f}, BACC: {train_bacc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}, Prec: {val_prec:.4f}, "
          f"Rec: {val_rec:.4f}, F1: {val_f1:.4f}, BACC: {val_bacc:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(resnet50.state_dict(), model_path)
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# =======================
# Save CSV
# =======================
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Train Prec', 'Train Rec', 'Train F1', 'Train BACC',
                     'Val Loss', 'Val Acc', 'Val Prec', 'Val Rec', 'Val F1', 'Val BACC'])
    for i in range(len(train_losses)):
        writer.writerow([i+1, train_losses[i], train_accs[i], train_precisions[i], train_recalls[i], train_f1s[i], train_baccs[i],
                         val_losses[i], val_accs[i], val_precisions[i], val_recalls[i], val_f1s[i], val_baccs[i]])
