import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parent / "new_dataset_15"
OUT_DIR      = Path(__file__).parent.parent / "results" / "figure2_class_distribution"
SPLITS       = ["train", "val", "test"]
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ── Count images ──────────────────────────────────────────────────────────────
def count_images(folder: Path) -> int:
    return sum(
        1 for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )

# Collect all class names from the union of all splits
all_classes: set[str] = set()
for split in SPLITS:
    split_dir = DATASET_ROOT / split
    if split_dir.exists():
        all_classes.update(d.name for d in split_dir.iterdir() if d.is_dir())

classes = sorted(all_classes)

# Build per-split and total counts
counts: dict[str, dict[str, int]] = {cls: {} for cls in classes}
for cls in classes:
    for split in SPLITS:
        folder = DATASET_ROOT / split / cls
        counts[cls][split] = count_images(folder) if folder.exists() else 0
    counts[cls]["total"] = sum(counts[cls][s] for s in SPLITS)

# ── Print table ───────────────────────────────────────────────────────────────
header = f"{'Class':<30} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>8}"
print(header)
print("-" * len(header))
for cls in classes:
    c = counts[cls]
    print(f"{cls:<30} {c['train']:>7} {c['val']:>7} {c['test']:>7} {c['total']:>8}")
grand = {s: sum(counts[cls][s] for cls in classes) for s in SPLITS + ["total"]}
print("-" * len(header))
print(f"{'TOTAL':<30} {grand['train']:>7} {grand['val']:>7} {grand['test']:>7} {grand['total']:>8}")

# ── Paper-quality bar chart ───────────────────────────────────────────────────
# Medium-contrast palette (accessible, not too saturated)
SPLIT_COLORS = {
    "train": "#4878CF",   # steel blue
    "val":   "#6ACC65",   # muted green
    "test":  "#D65F5F",   # muted red
}

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   15,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  13,
    "figure.dpi":       200,
    "savefig.dpi":      600,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

n      = len(classes)
x      = np.arange(n)
width  = 0.25
offsets = [-width, 0, width]

fig, ax = plt.subplots(figsize=(16, 7))

for i, (split, offset) in enumerate(zip(SPLITS, offsets)):
    vals = [counts[cls][split] for cls in classes]
    bars = ax.bar(
        x + offset, vals, width,
        label=split.capitalize(),
        color=SPLIT_COLORS[split],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    # Annotate bars with counts (skip if 0)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8,
                str(v),
                ha="center", va="bottom",
                fontsize=10, color="#333333",
            )

# Totals as a subtle line for quick reference
totals = [counts[cls]["total"] for cls in classes]
ax.plot(x, totals, color="#555555", marker="D", markersize=4,
        linewidth=1.2, linestyle="--", label="Total", zorder=4)

xtick_labels = classes
ax.set_xticks(x)
ax.set_xticklabels(xtick_labels, rotation=35, ha="right")

ax.set_ylabel("Number of Images")
ax.set_title("Class Distribution Across Train / Validation / Test Splits")
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()

out_path = OUT_DIR / "class_distribution.pdf"
plt.savefig(out_path, bbox_inches="tight")
print(f"\nSaved → {out_path}")

png_path = OUT_DIR / "class_distribution.png"
plt.savefig(png_path, bbox_inches="tight")
print(f"Saved → {png_path}")
plt.show()
