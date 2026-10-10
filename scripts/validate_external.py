"""
validate_external.py — Validate checkpoints/cgle.pth on external datasets (Gastrovision / Kvasir-v1).

The model was trained on 15 classes. This script finds the common classes between
the model's training classes and the external dataset, then evaluates only on those.
Argmax is restricted to common-class logits (closed-set recognition).

Usage:
    # Gastrovision
    python scripts/validate_external.py \
        --dataset /path/to/Gastrovision \
        --name Gastrovision

    # Kvasir-v1
    python scripts/validate_external.py \
        --dataset /path/to/kvasir-dataset-v1 \
        --name Kvasir

    # Just print matched classes without running inference
    python scripts/validate_external.py --dataset /path/to/dataset --name X --dry_run
"""

import os
import sys
import argparse
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cgle import LAG


# ── Class-name matching ────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Lowercase + collapse hyphens/underscores/spaces → single space."""
    return ' '.join(name.lower().replace('-', ' ').replace('_', ' ').split())


def match_classes(model_classes: list, dataset_folders: list) -> dict:
    """
    Fuzzy match: case-insensitive, hyphen/underscore-normalised.
    Returns {dataset_folder: (model_label_idx, model_class_name)}.
    """
    norm_to_model = {_norm(c): (i, c) for i, c in enumerate(model_classes)}
    matched = {}
    for folder in dataset_folders:
        key = _norm(folder)
        if key in norm_to_model:
            matched[folder] = norm_to_model[key]
    return matched


# ── Dataset ────────────────────────────────────────────────────────────────────

class FilteredFlatDataset(Dataset):
    """Loads images from a flat class-folder dataset, restricted to matched classes only."""

    _EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, root: str, folder_to_model_idx: dict, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        for folder, (model_idx, _) in sorted(folder_to_model_idx.items()):
            class_dir = os.path.join(root, folder)
            if not os.path.isdir(class_dir):
                print(f"  WARNING: folder not found, skipping: {class_dir}")
                continue
            files = [
                f for f in sorted(os.listdir(class_dir))
                if os.path.splitext(f)[1].lower() in self._EXTS
            ]
            for f in files:
                self.images.append(os.path.join(class_dir, f))
                self.labels.append(model_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── Inference (restricted to common classes) ───────────────────────────────────

def run_inference(model, loader, device, common_model_indices: list):
    """
    Argmax is taken only over logits of common_model_indices (closed-set eval).
    Returns (predictions_as_model_idx, true_labels_as_model_idx).
    """
    idx_tensor = torch.tensor(common_model_indices, dtype=torch.long, device=device)
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            logits = model(images)                        # (B, 15)
            restricted = logits[:, idx_tensor]            # (B, len(common))
            local_preds = torch.argmax(restricted, dim=1) # indices into common list
            preds = idx_tensor[local_preds]               # map back to model label space
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            if (i + 1) % 20 == 0:
                print(f"  Processed {(i+1) * loader.batch_size} / {len(loader.dataset)} images")

    return all_preds, all_labels


# ── Metrics & saving ───────────────────────────────────────────────────────────

def evaluate(preds, labels, idx_to_name: dict, dataset_name: str, out_dir: str):
    unique_labels = sorted(set(labels))
    names = [idx_to_name[i] for i in unique_labels]

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0, labels=unique_labels)
    rec  = recall_score(labels, preds, average='macro', zero_division=0, labels=unique_labels)
    f1   = f1_score(labels, preds, average='macro', zero_division=0, labels=unique_labels)

    sep = '=' * 65
    print(f"\n{sep}")
    print(f" Dataset : {dataset_name}   |   Common classes: {len(unique_labels)}")
    print(f"{sep}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (macro)")
    print(f"  Recall    : {rec:.4f}  (macro)")
    print(f"  F1 Score  : {f1:.4f}  (macro)")
    print()
    print(classification_report(
        labels, preds, labels=unique_labels, target_names=names, zero_division=0
    ))

    os.makedirs(out_dir, exist_ok=True)

    # Summary CSV
    pd.DataFrame([{
        'Dataset': dataset_name,
        'Common Classes': len(unique_labels),
        'Accuracy': round(acc, 4),
        'Precision (macro)': round(prec, 4),
        'Recall (macro)': round(rec, 4),
        'F1 (macro)': round(f1, 4),
    }]).to_csv(os.path.join(out_dir, 'metrics_summary.csv'), index=False)

    # Per-class CSV
    report_dict = classification_report(
        labels, preds, labels=unique_labels,
        target_names=names, zero_division=0, output_dict=True,
    )
    pd.DataFrame(report_dict).T.to_csv(os.path.join(out_dir, 'metrics_per_class.csv'))

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=unique_labels)
    n = len(unique_labels)
    fig_size = max(8, n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {dataset_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    print(f"Saved results to: {out_dir}/")
    return acc, prec, rec, f1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Validate model.pth on an external gastrointestinal dataset.'
    )
    parser.add_argument('--model', default='checkpoints/cgle.pth',
                        help='Path to model checkpoint (default: checkpoints/cgle.pth)')
    parser.add_argument('--dataset', required=True,
                        help='Path to external dataset root (class folders inside)')
    parser.add_argument('--name', required=True,
                        help='Dataset name used in output file names')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out_dir', default=None,
                        help='Output directory (default: results/external_validation/<name>)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print matched classes and exit without running inference')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    assert os.path.exists(args.model), f"Checkpoint not found: {args.model}"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)

    c2i = ckpt.get('class_to_idx')
    if c2i is None:
        print("ERROR: checkpoint is missing 'class_to_idx'. Cannot determine class names.")
        sys.exit(1)

    model_classes = [k for k, v in sorted(c2i.items(), key=lambda x: x[1])]
    print(f"\nModel trained on {len(model_classes)} classes:")
    for i, c in enumerate(model_classes):
        print(f"  [{i:2d}] {c}")

    # ── Find dataset folders ─────────────────────────────────────────────────
    dataset_root = args.dataset
    if not os.path.isdir(dataset_root):
        print(f"ERROR: dataset path does not exist: {dataset_root}")
        sys.exit(1)

    dataset_folders = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])
    print(f"\nExternal dataset '{args.name}' — {len(dataset_folders)} folders:")
    for f in dataset_folders:
        print(f"  {f}")

    # ── Match classes ────────────────────────────────────────────────────────
    matched = match_classes(model_classes, dataset_folders)

    if not matched:
        print("\nNo common classes found via normalised name matching.")
        print("Model classes (normalised) :", [_norm(c) for c in model_classes])
        print("Dataset folders (normalised):", [_norm(f) for f in dataset_folders])
        print("\nTip: use --dry_run to inspect names, then adjust folder names if needed.")
        sys.exit(1)

    print(f"\nMatched {len(matched)} common class(es):")
    print(f"  {'Dataset folder':<45}  →  model class [idx]")
    print(f"  {'-'*65}")
    for folder, (model_idx, model_cls) in sorted(matched.items()):
        count = len([
            f for f in os.listdir(os.path.join(dataset_root, folder))
            if os.path.splitext(f)[1].lower() in FilteredFlatDataset._EXTS
        ])
        print(f"  {folder:<45}  →  '{model_cls}' [{model_idx}]  ({count} images)")

    unmatched_model = [c for c in model_classes if _norm(c) not in {_norm(f) for f in matched}]
    unmatched_data  = [f for f in dataset_folders if f not in matched]
    if unmatched_model:
        print(f"\nModel classes with NO match in dataset ({len(unmatched_model)}):")
        for c in unmatched_model:
            print(f"  {c}")
    if unmatched_data:
        print(f"\nDataset folders with NO match in model ({len(unmatched_data)}):")
        for f in unmatched_data:
            print(f"  {f}")

    if args.dry_run:
        print("\n[dry_run] Exiting without running inference.")
        return

    # ── Load model ───────────────────────────────────────────────────────────
    model = LAG().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"\nLoaded epoch {ckpt.get('epoch', '?')}  "
          f"val_F1={ckpt.get('val_f1', float('nan')):.4f}")

    # ── Build DataLoader ─────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((720, 576)),   # same as training
        transforms.ToTensor(),
    ])

    dataset = FilteredFlatDataset(dataset_root, matched, transform=transform)
    print(f"\nDataset: {len(dataset)} images across {len(matched)} classes")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # ── Inference ────────────────────────────────────────────────────────────
    common_model_indices = sorted({idx for idx, _ in matched.values()})
    idx_to_name = {idx: name for idx, name in matched.values()}

    print("\nRunning inference...")
    preds, labels = run_inference(model, loader, device, common_model_indices)

    # ── Evaluate & save ──────────────────────────────────────────────────────
    out_dir = args.out_dir or os.path.join("results", "external_validation", args.name.lower().replace(" ", "_"))
    evaluate(preds, labels, idx_to_name, args.name, out_dir)


if __name__ == '__main__':
    main()
