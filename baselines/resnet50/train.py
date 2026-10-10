import os
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau

# =======================
# Hyperparameters
# =======================
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_CLASSES = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 224

# =======================
# Output Directory
# =======================
output_dir = os.path.join(ROOT, "runs", "resnet50")
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "resnet50_model_checkpoint.pth")
csv_path = os.path.join(output_dir, "training_log.csv")

# =======================
# Data Transforms
# =======================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

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
# Resume Checkpoint
# =======================
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
start_epoch = 0
best_val_loss = float("inf")
early_stop_counter = 0
patience = 50

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    resnet50.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    train_accuracies = checkpoint["train_accuracies"]
    val_accuracies = checkpoint["val_accuracies"]
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = min(val_losses)
    print(f"Resuming training from epoch {start_epoch}")

# =======================
# Training Loop
# =======================
for epoch in range(start_epoch, NUM_EPOCHS):
    resnet50.train()
    total_loss, total_correct = 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = resnet50(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()

    avg_train_loss = total_loss / len(train_loader)
    train_acc = 100 * total_correct / len(train_dataset)

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # Validation
    resnet50.eval()
    val_correct, val_loss_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = resnet50(images)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss_total / len(val_loader)
    val_acc = 100 * val_correct / len(val_dataset)

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    scheduler.step(avg_val_loss)

    print(f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")

    # Checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state": resnet50.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }
    torch.save(checkpoint, model_path)

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# =======================
# Save Plots
# =======================
plt.figure()
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "loss_curve.png"))

plt.figure()
plt.plot(train_accuracies, label='Train Accuracy', color='green')
plt.plot(val_accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))

# =======================
# Save CSV
# =======================
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])
    for i in range(len(train_losses)):
        writer.writerow([i + 1, train_losses[i], train_accuracies[i], val_losses[i], val_accuracies[i]])
