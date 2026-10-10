import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Folder with one sub-folder per class, each holding <img>.jpg and score_cam_<img>.jpg
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "figure4_scorecam", "panels")
OUT_DIR = os.path.join(ROOT, "..")

# Collect subdirs in sorted order
subdirs = sorted([
    d for d in os.listdir(ROOT)
    if os.path.isdir(os.path.join(ROOT, d))
])

raw_images, scorecam_images, labels = [], [], []

for subdir in subdirs:
    subdir_path = os.path.join(ROOT, subdir)
    all_jpgs = glob.glob(os.path.join(subdir_path, "*.jpg"))

    score_cam = [p for p in all_jpgs if os.path.basename(p).startswith("score_cam_")]
    raw = [p for p in all_jpgs if not os.path.basename(p).startswith("score_cam_")]

    assert len(raw) == 1 and len(score_cam) == 1, \
        f"Expected exactly 1 raw and 1 score-cam image in {subdir}"

    raw_images.append(raw[0])
    scorecam_images.append(score_cam[0])

    # Format label: strip leading index, keep dataset folder name as-is
    label = subdir.split("_", 1)[-1]
    labels.append(label)

# ── Figure layout ─────────────────────────────────────────────────────────────
N = len(subdirs)

# Size each cell to the image aspect ratio so imshow fills it without letterboxing.
_s = mpimg.imread(raw_images[0])
_aspect = _s.shape[0] / _s.shape[1]  # height / width

CELL_W = 3.2
FIG_W  = N * CELL_W
CELL_H = CELL_W * _aspect
CBAR_H = 0.25   # inches reserved for the colorbar strip
FIG_H  = 2 * CELL_H + 0.8 + CBAR_H  # +0.8" titles/labels, +CBAR_H for scale meter

# Equal absolute gap: wspace × W_ax = hspace × H_ax
WSPACE = 0.04
HSPACE = WSPACE * CELL_W / CELL_H

fig, axes = plt.subplots(
    2, N,
    figsize=(FIG_W, FIG_H),
    gridspec_kw={"hspace": HSPACE, "wspace": WSPACE},
)

row_labels = ["Original", "Score-CAM"]

for col, (raw_path, sc_path, label) in enumerate(
    zip(raw_images, scorecam_images, labels)
):
    for row, img_path in enumerate([raw_path, sc_path]):
        ax = axes[row, col]
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Column title on top row
        if row == 0:
            ax.set_title(label, fontsize=12, fontweight="bold", pad=8)

# Row labels on the left column
for row, rl in enumerate(row_labels):
    axes[row, 0].set_ylabel(
        rl, fontsize=14, fontweight="bold",
        rotation=90, labelpad=10, va="center"
    )

# ── Activation scale meter (colorbar) ─────────────────────────────────────────
# Score-CAM overlays use the jet colormap; 0 = no activation, 1 = peak activation.
# The colorbar is placed below the grid as a shared horizontal scale for all columns.
sm = plt.cm.ScalarMappable(
    cmap="jet",
    norm=plt.Normalize(vmin=0, vmax=1),
)
sm.set_array([])

# Shrink the colorbar to 60 % of figure width and center it
cbar = fig.colorbar(
    sm,
    ax=axes,
    orientation="horizontal",
    fraction=0.02,
    pad=0.04,
    shrink=0.60,
    aspect=40,
)
cbar.set_label("Activation Score (Score-CAM)", fontsize=13, labelpad=6)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["Low", "", "Medium", "", "High"], fontsize=12)

out_png = os.path.join(OUT_DIR, "scorecam_figure.png")
out_pdf = os.path.join(OUT_DIR, "scorecam_figure.pdf")

fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved:\n  {out_png}\n  {out_pdf}")
