"""
Train CGLE (or one of its ablation variants) on the 15-class HyperKvasir subset.

    python scripts/train.py                              # full CGLE model
    python scripts/train.py --model global_only --out runs/global_only
    python scripts/train.py --model local_only  --out runs/local_only
    python scripts/train.py --model fusion_no_ca --out runs/fusion_no_ca

Outputs (under --out):
    checkpoints/model.pth       latest epoch (training resumes from it if present)
    checkpoints/best_model.pth  best validation macro-F1
    logs/metrics_log.csv        per-epoch train/val metrics
    plots/                      loss/accuracy curves, validation confusion matrix
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cgle import MODELS
from cgle.data import preprocess_dataset

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", choices=sorted(MODELS), default="cgle")
parser.add_argument("--data", default="data/new_dataset_15", help="dataset root with train/ val/ test/")
parser.add_argument("--out", default="runs/cgle", help="output directory")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=50, help="early-stopping patience on val loss")
args = parser.parse_args()

# ------------------- Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = MODELS[args.model]().to(device)
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.3, patience=2, min_lr=1e-6
)

# ------------------- Paths -------------------
root_dir = args.out
ckpt_dir = os.path.join(root_dir, "checkpoints")
logs_dir = os.path.join(root_dir, "logs")
plots_dir = os.path.join(root_dir, "plots")
for d in (ckpt_dir, logs_dir, plots_dir):
    os.makedirs(d, exist_ok=True)

model_path      = os.path.join(ckpt_dir, "model.pth")       # latest (for resume)
best_model_path = os.path.join(ckpt_dir, "best_model.pth")  # best val F1
csv_path        = os.path.join(logs_dir, "metrics_log.csv")
loss_plot_path  = os.path.join(plots_dir, "loss_curve.png")
acc_plot_path   = os.path.join(plots_dir, "accuracy_curve.png")

# ------------------- Load Data -------------------
data = preprocess_dataset(root_dir=args.data)
train_loader = data.train_dataloader
val_loader   = data.validation_dataloader
CLASS_TO_IDX = data.training_dataset.class_to_idx
print(f"Class order: {[k for k, v in sorted(CLASS_TO_IDX.items(), key=lambda x: x[1])]}")

# ------------------- Resume Checkpoint -------------------
train_losses, val_losses = [], []
train_accuracy, val_accuracy = [], []
metrics_list = []
start_epoch = 1
best_val_loss = float("inf")
best_val_f1   = 0.0
early_stop_counter = 0
patience = args.patience

# Load previous CSV metrics if exists
if os.path.exists(csv_path):
    old_df = pd.read_csv(csv_path)
    metrics_list = old_df.to_dict('records')
    start_epoch = old_df['Epoch'].max() + 1
    print(f"Resuming metrics logging from epoch {start_epoch}")

# Load checkpoint if exists
if os.path.exists(model_path):
    print("Checkpoint found. Loading...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_losses  = checkpoint.get("train_loss", [])
    val_losses    = checkpoint.get("valid_loss", [])
    train_accuracy = checkpoint.get("t_accuracy", [])
    val_accuracy  = checkpoint.get("v_accuracy", [])
    best_val_loss = min(val_losses) if val_losses else float("inf")
    best_val_f1   = checkpoint.get("best_val_f1", 0.0)
    print(f"Resumed from epoch {start_epoch}")

# ------------------- Training Loop -------------------
total_epochs = args.epochs
for epoch in range(start_epoch, total_epochs + 1):
    model.train()
    s_time = time.time()
    train_loss = 0.0
    val_epoch_loss = 0.0
    train_preds, train_labels = [], []
    val_preds, val_labels = [], []

    # --- Training ---
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{total_epochs}] Training", leave=False)
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels.long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        train_loss += loss.item()
        train_preds += torch.argmax(outputs, dim=1).cpu().tolist()
        train_labels += labels.cpu().tolist()

        train_bar.set_postfix(loss=loss.item())

    # --- Validation ---
    model.eval()
    val_bar = tqdm(val_loader, desc=f"Epoch [{epoch}/{total_epochs}] Validation", leave=False)
    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels.long())

            val_epoch_loss += loss.item()
            val_preds += torch.argmax(outputs, dim=1).cpu().tolist()
            val_labels += labels.cpu().tolist()

            val_bar.set_postfix(loss=loss.item())

    # --- Metrics ---
    train_loss /= len(train_loader)
    val_epoch_loss /= len(val_loader)

    # Main metrics
    train_acc = accuracy_score(train_labels, train_preds)
    val_acc = accuracy_score(val_labels, val_preds)

    # Additional metrics
    train_precision = precision_score(train_labels, train_preds, average='macro', zero_division=0)
    train_recall = recall_score(train_labels, train_preds, average='macro', zero_division=0)
    train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    train_bacc = balanced_accuracy_score(train_labels, train_preds)

    val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    val_bacc = balanced_accuracy_score(val_labels, val_preds)

    scheduler.step(val_epoch_loss)

    train_losses.append(train_loss)
    val_losses.append(val_epoch_loss)
    train_accuracy.append(train_acc)
    val_accuracy.append(val_acc)

    print(f"\nEpoch [{epoch}/{total_epochs}] - Time: {time.time() - s_time:.2f}s")
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, "
          f"F1: {train_f1:.4f}, BACC: {train_bacc:.4f}")
    print(f"Val   Loss: {val_epoch_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
          f"F1: {val_f1:.4f}, BACC: {val_bacc:.4f}")

    # --- Save Checkpoint (latest, used for resume) ---
    ckpt_payload = {
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss':           train_losses,
        'valid_loss':           val_losses,
        't_accuracy':           train_accuracy,
        'v_accuracy':           val_accuracy,
        'val_f1':               val_f1,
        'best_val_f1':          best_val_f1,
        'class_to_idx':         CLASS_TO_IDX,
    }
    torch.save(ckpt_payload, model_path)

    # --- Save Best Model (by val F1) ---
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        ckpt_payload['best_val_f1'] = best_val_f1
        torch.save(ckpt_payload, best_model_path)
        print(f"  ✓ Best model saved (val_F1={best_val_f1:.4f})")

    # --- Append Metrics ---
    metrics_list.append({
        'Epoch': epoch,
        'Train Loss': train_loss,
        'Val Loss': val_epoch_loss,
        'Train Accuracy': train_acc,
        'Val Accuracy': val_acc,
        'Train Precision': train_precision,
        'Val Precision': val_precision,
        'Train Recall': train_recall,
        'Val Recall': val_recall,
        'Train F1': train_f1,
        'Val F1': val_f1,
        'Train BACC': train_bacc,
        'Val BACC': val_bacc,
    })

    pd.DataFrame(metrics_list).to_csv(csv_path, index=False)

    # --- Early Stopping ---
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"EarlyStopping counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# ------------------- Save Curves -------------------
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Loss Curve")
plt.savefig(loss_plot_path)
plt.close()

plt.figure()
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(acc_plot_path)
plt.close()

# ------------------- Confusion Matrix -------------------
print("\n=== Confusion Matrix (Validation Set) ===")
cm = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
plt.close()
print("Confusion matrix saved at:", os.path.join(plots_dir, "confusion_matrix.png"))
