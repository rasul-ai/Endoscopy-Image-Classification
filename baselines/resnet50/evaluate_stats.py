"""
stat_resnet/infer.py  —  ResNet-50 baseline (journal-ready statistical validation)
====================================================================================
Produces all three validation blocks required by reviewers:

  Block 1 – Discrimination
    • Accuracy, Precision, Recall (Sensitivity), Specificity, F1  +  95% CI
    • Macro AUC (OvR) + DeLong 95% CI (per class + macro average)
    • Bootstrap 95% CI (all metrics)
    → paper_table_discrimination.csv, metrics_discrimination.csv,
      delong_per_class_auc.csv, per_class_sens_spec.csv

  Block 2 – Calibration  (clinical reliability)
    • Brier Score + Calibration Slope + Calibration Intercept
    • Hosmer–Lemeshow test
    • Reliability diagrams
    → paper_table_calibration.csv, calibration_detailed.csv,
      calibration_reliability.png

  Block 3 – Clinical Utility  (DCA)
    • Per-class Decision Curve plots
    • Combined macro DCA
    • Net benefit table at key thresholds
    → dca.png, dca_combined_macro.png, dca_net_benefit_table.csv

  Plus: confusion matrix, ROC curves, scores.npz

All outputs → results/per_model/resnet50/

Usage
-----
  python baselines/resnet50/evaluate_stats.py   # from the repository root
"""

import os
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# ── import shared utilities from Stat/ ────────────────────────────
sys.path.insert(0, os.path.join(ROOT, "statistics"))
from stat_utils import (
    macro_auc_with_delong_ci,
    bootstrap_ci,
    per_class_sens_spec,
    calibration_analysis,
    decision_curve_analysis,
    plot_roc_curves,
    plot_confusion_matrix,
    delong_vs_reference,
    print_summary,
    save_metrics_csv,
    save_delong_vs_ref_csv,
    macro_calibration_stats,
    macro_delong_comparison,
    plot_combined_dca_macro,
    save_paper_discrimination_table,
    save_paper_calibration_table,
)

# ─────────────────────────── Config ───────────────────────────────
VARIANT_NAME = "ResNet-50"
NUM_CLASSES  = 15
IMAGE_SIZE   = 224
BATCH_SIZE   = 16
N_BOOT       = 1000

TRAIN_ROOT   = os.path.join(ROOT, "data", "new_dataset_15", "train")
TEST_ROOT    = os.path.join(ROOT, "data", "new_dataset_15", "test")
MODEL_PATH   = os.path.join(ROOT, "checkpoints", "resnet50.pth")
OUT_DIR      = os.path.join(ROOT, "results", "per_model", "resnet50")
SCORES_NPZ   = os.path.join(OUT_DIR, "scores.npz")

os.makedirs(OUT_DIR, exist_ok=True)

# Comparison model score paths (add paths to other model scores.npz files here
# to enable DeLong paired significance tests and calibration comparisons).
# Leave empty to skip cross-model comparison tables.
COMPARISON_SCORES: dict[str, str] = {
    # Example:
    # "LAG (Full)": os.path.normpath(
    #     os.path.join(os.path.dirname(__file__), "..",
    #                  "Stat", "results", "full", "scores.npz")),
}

# ─────────────────────────── Device ───────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─────────────────────────── Class names ──────────────────────────
# ResNet-50 checkpoints in this project do not store class_to_idx,
# so we derive the canonical class order from the training folder.
assert os.path.isdir(TRAIN_ROOT), (
    f"Training root not found: {TRAIN_ROOT}\n"
    "Set TRAIN_ROOT to the correct path.")
_train_ref = datasets.ImageFolder(root=TRAIN_ROOT)
CLASS_NAMES = _train_ref.classes          # sorted alphabetically by ImageFolder
N_CLASSES   = len(CLASS_NAMES)
assert N_CLASSES == NUM_CLASSES, (
    f"Expected {NUM_CLASSES} classes, found {N_CLASSES} in {TRAIN_ROOT}")
print(f"Classes ({N_CLASSES}): {CLASS_NAMES}")

# ─────────────────────────── Data ─────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

assert os.path.isdir(TEST_ROOT), f"Test root not found: {TEST_ROOT}"
test_dataset = datasets.ImageFolder(root=TEST_ROOT, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
print(f"Test samples: {len(test_dataset)}")

# ─────────────────────────── Model ────────────────────────────────
assert os.path.exists(MODEL_PATH), f"No checkpoint found: {MODEL_PATH}"
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

_epoch = ckpt.get("epoch", "?")
_val_accs = ckpt.get("val_accuracies", [])
_best_val  = max(_val_accs) if _val_accs else float("nan")
print(f"Loaded checkpoint  epoch={_epoch}  best_val_acc={_best_val:.2f}%")

# ─────────────────────────── Inference ────────────────────────────
all_probs, all_preds, all_labels = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1).tolist()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_prob = np.array(all_probs)      # shape: (N, n_classes)

np.savez(SCORES_NPZ, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
print(f"Scores saved → {SCORES_NPZ}\n")


# ══════════════════════════════════════════════════════════════════
#  BLOCK 1 — DISCRIMINATION
# ══════════════════════════════════════════════════════════════════
print("═" * 62)
print("  BLOCK 1 — DISCRIMINATION")
print("═" * 62)

kw         = dict(average="macro", zero_division=0)
acc        = accuracy_score(y_true, y_pred)
bacc       = balanced_accuracy_score(y_true, y_pred)
precision  = precision_score(y_true, y_pred, **kw)
f1         = f1_score(y_true, y_pred, **kw)
sens_cls, spec_cls = per_class_sens_spec(y_true, y_pred, N_CLASSES)
sens_macro = float(np.mean(sens_cls))
spec_macro = float(np.mean(spec_cls))

y_bin     = label_binarize(y_true, classes=list(range(N_CLASSES)))
macro_auc = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))

print("\nComputing DeLong 95% CI …")
_, delong_per_class = macro_auc_with_delong_ci(y_true, y_prob, N_CLASSES, CLASS_NAMES)
delong_ci_low  = float(np.nanmean([v["ci_low"]  for v in delong_per_class.values()]))
delong_ci_high = float(np.nanmean([v["ci_high"] for v in delong_per_class.values()]))

print(f"Computing Bootstrap CI ({N_BOOT} samples) …")
ci = bootstrap_ci(y_true, y_pred, y_prob, N_CLASSES, n_boot=N_BOOT)

disc_metrics = {
    "Accuracy":                        acc,
    "Accuracy 95% CI (Bootstrap)":     ci["acc"],
    "Balanced Accuracy":               bacc,
    "Precision (macro)":               precision,
    "Sensitivity / Recall (macro)":    sens_macro,
    "Sensitivity 95% CI (Bootstrap)":  ci["sens"],
    "Specificity (macro)":             spec_macro,
    "Specificity 95% CI (Bootstrap)":  ci["spec"],
    "F1 Score (macro)":                f1,
    "F1 95% CI (Bootstrap)":           ci["f1"],
    "AUC (macro, OvR)":                macro_auc,
    "AUC 95% CI (DeLong, macro)":      (delong_ci_low, delong_ci_high),
    "AUC 95% CI (Bootstrap)":          ci["auc"],
}
print_summary(disc_metrics, VARIANT_NAME)
save_metrics_csv(disc_metrics, OUT_DIR, filename="metrics_discrimination.csv")

# Per-class DeLong AUC table
delong_rows = [
    {"Class": cls, "AUC": v["auc"], "CI_low": v["ci_low"], "CI_high": v["ci_high"],
     "CI_str": f"{v['ci_low']:.3f}–{v['ci_high']:.3f}"}
    for cls, v in delong_per_class.items()
]
pd.DataFrame(delong_rows).to_csv(
    os.path.join(OUT_DIR, "delong_per_class_auc.csv"), index=False)

# Per-class sensitivity / specificity
pd.DataFrame({"Class": CLASS_NAMES, "Sensitivity": sens_cls,
              "Specificity": spec_cls}).to_csv(
    os.path.join(OUT_DIR, "per_class_sens_spec.csv"), index=False)

# Paper discrimination table  (reference model + any comparisons)
disc_table_rows = [{
    "Model":           VARIANT_NAME,
    "AUC":             f"{macro_auc:.3f}",
    "95% CI (DeLong)": f"{delong_ci_low:.3f}–{delong_ci_high:.3f}",
    "p-value vs Ours": "—",
}]

for vname, vpath in COMPARISON_SCORES.items():
    if not os.path.exists(vpath):
        print(f"  [skip] {vname}: scores not found ({vpath})")
        continue
    vdata = np.load(vpath)
    vprob = vdata["y_prob"]
    delong_res, delong_summary = delong_vs_reference(
        y_true, vprob, y_prob, N_CLASSES, CLASS_NAMES)
    safe_name = (vname.replace(" ", "_").replace("—", "")
                      .replace("(", "").replace(")", ""))
    save_delong_vs_ref_csv(delong_res, OUT_DIR,
                           filename=f"delong_vs_{safe_name}.csv")
    print(f"\n  DeLong [{VARIANT_NAME}] vs [{vname}]")
    print(f"    Bonferroni significant: "
          f"{delong_summary['n_significant_bonf']} / "
          f"{delong_summary['n_classes_tested']} classes")
    _, comp_delong = macro_auc_with_delong_ci(y_true, vprob, N_CLASSES, CLASS_NAMES)
    comp_auc   = float(np.nanmean([v["auc"]     for v in comp_delong.values()]))
    comp_ci_lo = float(np.nanmean([v["ci_low"]  for v in comp_delong.values()]))
    comp_ci_hi = float(np.nanmean([v["ci_high"] for v in comp_delong.values()]))
    _, _, _, p_macro = macro_delong_comparison(
        y_true, vprob, y_prob, N_CLASSES, CLASS_NAMES)
    disc_table_rows.append({
        "Model":           vname,
        "AUC":             f"{comp_auc:.3f}",
        "95% CI (DeLong)": f"{comp_ci_lo:.3f}–{comp_ci_hi:.3f}",
        "p-value vs Ours": f"{p_macro:.4f}" if not np.isnan(p_macro) else "N/A",
    })

save_paper_discrimination_table(disc_table_rows, OUT_DIR)
print("\nPaper Discrimination Table:")
print(pd.DataFrame(disc_table_rows).to_string(index=False))


# ══════════════════════════════════════════════════════════════════
#  BLOCK 2 — CALIBRATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("  BLOCK 2 — CALIBRATION")
print("═" * 62)

print("\nRunning reliability diagram + ECE + Brier …")
cal = calibration_analysis(y_true, y_prob, CLASS_NAMES, OUT_DIR, VARIANT_NAME)

print("Computing calibration slope, intercept, Hosmer–Lemeshow test …")
cal_stats = macro_calibration_stats(y_true, y_prob, N_CLASSES, CLASS_NAMES)

w = 38
print(f"\n  {'Metric':<{w}} Value")
print(f"  {'─' * 50}")
print(f"  {'Macro Brier Score':<{w}} {cal['macro_brier']:.4f}  (lower = better)")
print(f"  {'Macro ECE':<{w}} {cal['macro_ece']:.4f}")
print(f"  {'Macro Calibration Slope':<{w}} {cal_stats['macro_slope']:.4f}  (ideal = 1)")
print(f"  {'Macro Calibration Intercept':<{w}} {cal_stats['macro_intercept']:.4f}  (ideal = 0)")
print(f"  {'Hosmer–Lemeshow p (macro avg)':<{w}} {cal_stats['macro_hl_p']:.4f}  (> 0.05 = well-calibrated)")
print(f"  Reliability diagram → {cal['fig_path']}")

cal_detail_rows = []
for cls in CLASS_NAMES:
    pc = cal_stats["per_class"].get(cls, {})
    cal_detail_rows.append({
        "Class":         cls,
        "Brier_Score":   cal["per_class_brier"].get(cls, np.nan),
        "ECE":           cal["per_class_ece"].get(cls, np.nan),
        "Cal_Slope":     pc.get("slope",     np.nan),
        "Cal_Intercept": pc.get("intercept", np.nan),
        "HL_chi2":       pc.get("hl_chi2",   np.nan),
        "HL_p_value":    pc.get("hl_p",      np.nan),
    })
cal_detail_path = os.path.join(OUT_DIR, "calibration_detailed.csv")
pd.DataFrame(cal_detail_rows).to_csv(cal_detail_path, index=False)
print(f"  Per-class calibration details → {cal_detail_path}")

cal_table_rows = [{
    "Model":          VARIANT_NAME,
    "Brier Score":    f"{cal['macro_brier']:.3f}",
    "Cal. Slope":     f"{cal_stats['macro_slope']:.3f}",
    "Cal. Intercept": f"{cal_stats['macro_intercept']:.3f}",
    "HL p-value":     f"{cal_stats['macro_hl_p']:.3f}",
}]

for vname, vpath in COMPARISON_SCORES.items():
    if not os.path.exists(vpath):
        continue
    vdata = np.load(vpath)
    vprob = vdata["y_prob"]
    vtrue = vdata["y_true"] if "y_true" in vdata else y_true
    vcal  = calibration_analysis(vtrue, vprob, CLASS_NAMES,
                                 os.path.dirname(vpath), vname)
    vcs   = macro_calibration_stats(vtrue, vprob, N_CLASSES, CLASS_NAMES)
    cal_table_rows.append({
        "Model":          vname,
        "Brier Score":    f"{vcal['macro_brier']:.3f}",
        "Cal. Slope":     f"{vcs['macro_slope']:.3f}",
        "Cal. Intercept": f"{vcs['macro_intercept']:.3f}",
        "HL p-value":     f"{vcs['macro_hl_p']:.3f}",
    })

save_paper_calibration_table(cal_table_rows, OUT_DIR)
print("\nPaper Calibration Table:")
print(pd.DataFrame(cal_table_rows).to_string(index=False))


# ══════════════════════════════════════════════════════════════════
#  BLOCK 3 — CLINICAL UTILITY  (DCA)
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("  BLOCK 3 — CLINICAL UTILITY (DCA)")
print("═" * 62)

print("\nRunning per-class Decision Curve Analysis …")
dca = decision_curve_analysis(y_true, y_prob, CLASS_NAMES, OUT_DIR, VARIANT_NAME)
print(f"  Per-class DCA figure  → {dca['fig_path']}")

models_for_dca = {
    VARIANT_NAME: {"y_true": y_true, "y_prob": y_prob, "n_classes": N_CLASSES}
}
for vname, vpath in COMPARISON_SCORES.items():
    if os.path.exists(vpath):
        vd = np.load(vpath)
        models_for_dca[vname] = {
            "y_true":    vd["y_true"] if "y_true" in vd else y_true,
            "y_prob":    vd["y_prob"],
            "n_classes": N_CLASSES,
        }

dca_combined_path = plot_combined_dca_macro(
    models_for_dca, OUT_DIR,
    title=f"Decision Curve Analysis — {VARIANT_NAME}")
print(f"  Combined macro DCA    → {dca_combined_path}")

thresholds_key = [0.1, 0.2, 0.3, 0.4, 0.5]
y_bin_dca  = label_binarize(y_true, classes=list(range(N_CLASSES)))
n_samp     = len(y_true)
prev_macro = float(y_bin_dca.mean(axis=0).mean())

dca_nb_rows = []
for pt in thresholds_key:
    row = {"Threshold": pt}
    for mname, mdata in models_for_dca.items():
        y_p_m = mdata["y_prob"]
        nb = 0.0
        for c in range(N_CLASSES):
            pred_pos = y_p_m[:, c] >= pt
            tp = float(np.sum(pred_pos & (y_bin_dca[:, c] == 1)))
            fp = float(np.sum(pred_pos & (y_bin_dca[:, c] == 0)))
            nb += tp / n_samp - fp / n_samp * (pt / (1.0 - pt))
        row[mname] = round(nb / N_CLASSES, 4)
    row["Treat-All"]  = round(prev_macro - (1.0 - prev_macro) * pt / (1.0 - pt), 4)
    row["Treat-None"] = 0.0
    dca_nb_rows.append(row)

dca_nb_path = os.path.join(OUT_DIR, "dca_net_benefit_table.csv")
pd.DataFrame(dca_nb_rows).to_csv(dca_nb_path, index=False)
print(f"  Net benefit table     → {dca_nb_path}")
print("\nNet Benefit at Key Thresholds:")
print(pd.DataFrame(dca_nb_rows).to_string(index=False))


# ══════════════════════════════════════════════════════════════════
#  ADDITIONAL OUTPUTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("  ADDITIONAL OUTPUTS")
print("═" * 62)

cm_path  = plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, OUT_DIR, VARIANT_NAME)
print(f"\n  Confusion matrix → {cm_path}")
roc_path = plot_roc_curves(y_true, y_prob, CLASS_NAMES, OUT_DIR, VARIANT_NAME)
print(f"  ROC curves       → {roc_path}")

print(f"\n{'─' * 62}")
print(f"  All outputs saved to: {OUT_DIR}")
print(f"{'─' * 62}\n")
