"""
stat_divgi/infer.py  —  DivGI Single-Label model  (journal-ready statistical validation)
==========================================================================================
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

All outputs → results/per_model/divgi/

Usage: python baselines/divgi/evaluate_stats.py   # from the repository root
"""

import os
import sys

# DivGI code (models/, utils/) lives next to this file; shared stats in statistics/
_DIVGI_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_DIVGI_DIR, "..", ".."))
sys.path.insert(0, _DIVGI_DIR)
sys.path.insert(0, os.path.join(ROOT, "statistics"))

import numpy as np
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from models.get_model import get_arch
from utils.reproducibility import set_seeds
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
VARIANT_NAME = "DivGI — Full Model"

# Paths are resolved relative to the repo root so the script can be
# launched from any working directory.
MODEL_PATH = os.path.join(ROOT, "checkpoints", "divgi.pth")
DATA_PATH  = os.path.join(ROOT, "data", "new_dataset_15")
OUT_DIR    = os.path.join(ROOT, "results", "per_model", "divgi")
SCORES_NPZ = os.path.join(OUT_DIR, "scores.npz")

NUM_CLASSES          = 15
BATCH_SIZE           = 32
DO_MULTIGRANULARITIES = True
N_STEP               = 4
N_BOOT               = 1000

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────── Device & Reproducibility ─────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
set_seeds(42, torch.cuda.is_available())

# ─────────────────────────── Build & Load model ───────────────────
assert os.path.exists(MODEL_PATH), (
    f"Checkpoint not found: {MODEL_PATH}\n"
    "Set MODEL_PATH at the top of this script to point to best_model.pth "
    "or best_divgl.pth."
)

print(f"Loading DivGI model from: {MODEL_PATH}")
model, mean, std = get_arch(
    model_name="resnet50",
    n_classes=NUM_CLASSES,
    do_multigranularities=DO_MULTIGRANULARITIES,
    n_step=N_STEP,
    do_multilabel=False,
    number_multi_label=[NUM_CLASSES],
)
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()
print("Model loaded successfully.\n")

# ─────────────────────────── Data ─────────────────────────────────
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

test_dataset = datasets.ImageFolder(root=os.path.join(DATA_PATH, "test"),
                                    transform=transform_test)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)

CLASS_NAMES = test_dataset.classes          # sorted alphabetically by ImageFolder
N_CLASSES   = len(CLASS_NAMES)
assert N_CLASSES == NUM_CLASSES, (
    f"Expected {NUM_CLASSES} classes but dataset has {N_CLASSES}. "
    "Update NUM_CLASSES at the top of this script."
)
print(f"Classes ({N_CLASSES}): {CLASS_NAMES}\n")

# ─────────────────────────── Inference ────────────────────────────
print("Running inference on test set …")
all_probs, all_preds, all_labels = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)

        outputs = model(inputs, stage=1)          # full forward pass

        if DO_MULTIGRANULARITIES:
            # outputs[0] is a list of per-granularity logit tensors
            logits_list = outputs[0]
            logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        else:
            logits = outputs[0][-1]

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1).tolist()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_prob = np.array(all_probs)          # shape (N, n_classes)

np.savez(SCORES_NPZ, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
print(f"Scores saved → {SCORES_NPZ}\n")


# ══════════════════════════════════════════════════════════════════
#  BLOCK 1 — DISCRIMINATION
# ══════════════════════════════════════════════════════════════════
print("═" * 62)
print("  BLOCK 1 — DISCRIMINATION")
print("═" * 62)

kw        = dict(average="macro", zero_division=0)
acc       = accuracy_score(y_true, y_pred)
bacc      = balanced_accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, **kw)
f1        = f1_score(y_true, y_pred, **kw)

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
    {
        "Class":  cls,
        "AUC":    v["auc"],
        "CI_low": v["ci_low"],
        "CI_high": v["ci_high"],
        "CI_str": f"{v['ci_low']:.3f}–{v['ci_high']:.3f}",
    }
    for cls, v in delong_per_class.items()
]
pd.DataFrame(delong_rows).to_csv(
    os.path.join(OUT_DIR, "delong_per_class_auc.csv"), index=False)

# Per-class sensitivity / specificity
pd.DataFrame({
    "Class":       CLASS_NAMES,
    "Sensitivity": sens_cls,
    "Specificity": spec_cls,
}).to_csv(os.path.join(OUT_DIR, "per_class_sens_spec.csv"), index=False)

# Paper discrimination table — DivGI is the reference; no ablation variants here
disc_table_rows = [{
    "Model":           VARIANT_NAME,
    "AUC":             f"{macro_auc:.3f}",
    "95% CI (DeLong)": f"{delong_ci_low:.3f}–{delong_ci_high:.3f}",
    "Accuracy":        f"{acc:.3f}",
    "F1 (macro)":      f"{f1:.3f}",
    "Sensitivity":     f"{sens_macro:.3f}",
    "Specificity":     f"{spec_macro:.3f}",
    "p-value vs Ref":  "—",
}]

# ── Optional: compare against a reference model score file ──────
# If you have another model's scores.npz (e.g. a comparison baseline),
# set REFERENCE_SCORES to its path; otherwise leave as None.
REFERENCE_SCORES: str | None = None          # e.g. "../Stat/results/lag/scores.npz"

if REFERENCE_SCORES and os.path.exists(REFERENCE_SCORES):
    ref_data = np.load(REFERENCE_SCORES)
    ref_prob = ref_data["y_prob"]
    print("\nRunning DeLong comparison vs reference …")
    delong_res, delong_summary = delong_vs_reference(
        y_true, ref_prob, y_prob, N_CLASSES, CLASS_NAMES
    )
    save_delong_vs_ref_csv(delong_res, OUT_DIR, filename="delong_vs_reference.csv")
    print(f"  Bonferroni significant: "
          f"{delong_summary['n_significant_bonf']} / "
          f"{delong_summary['n_classes_tested']} classes")
    _, ref_delong = macro_auc_with_delong_ci(y_true, ref_prob, N_CLASSES, CLASS_NAMES)
    ref_auc   = float(np.nanmean([v["auc"]     for v in ref_delong.values()]))
    ref_ci_lo = float(np.nanmean([v["ci_low"]  for v in ref_delong.values()]))
    ref_ci_hi = float(np.nanmean([v["ci_high"] for v in ref_delong.values()]))
    _, _, _, p_macro = macro_delong_comparison(
        y_true, ref_prob, y_prob, N_CLASSES, CLASS_NAMES
    )
    disc_table_rows.append({
        "Model":           "Reference",
        "AUC":             f"{ref_auc:.3f}",
        "95% CI (DeLong)": f"{ref_ci_lo:.3f}–{ref_ci_hi:.3f}",
        "Accuracy":        "—",
        "F1 (macro)":      "—",
        "Sensitivity":     "—",
        "Specificity":     "—",
        "p-value vs Ref":  f"{p_macro:.4f}" if not np.isnan(p_macro) else "N/A",
    })
else:
    if REFERENCE_SCORES:
        print(f"  [skip] Reference scores not found: {REFERENCE_SCORES}")

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

if REFERENCE_SCORES and os.path.exists(REFERENCE_SCORES):
    ref_data  = np.load(REFERENCE_SCORES)
    ref_prob  = ref_data["y_prob"]
    ref_true  = ref_data["y_true"] if "y_true" in ref_data else y_true
    vcal  = calibration_analysis(ref_true, ref_prob, CLASS_NAMES,
                                 os.path.dirname(REFERENCE_SCORES), "Reference")
    vcs   = macro_calibration_stats(ref_true, ref_prob, N_CLASSES, CLASS_NAMES)
    cal_table_rows.append({
        "Model":          "Reference",
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
print(f"  Per-class DCA figure → {dca['fig_path']}")

models_for_dca = {
    VARIANT_NAME: {"y_true": y_true, "y_prob": y_prob, "n_classes": N_CLASSES}
}
if REFERENCE_SCORES and os.path.exists(REFERENCE_SCORES):
    rd = np.load(REFERENCE_SCORES)
    models_for_dca["Reference"] = {
        "y_true":    rd["y_true"] if "y_true" in rd else y_true,
        "y_prob":    rd["y_prob"],
        "n_classes": N_CLASSES,
    }

dca_combined_path = plot_combined_dca_macro(
    models_for_dca, OUT_DIR,
    title=f"Decision Curve Analysis — {VARIANT_NAME}",
)
print(f"  Combined macro DCA   → {dca_combined_path}")

thresholds_key = [0.1, 0.2, 0.3, 0.4, 0.5]
n_samp         = len(y_true)
prev_macro     = float(y_bin.mean(axis=0).mean())

dca_nb_rows = []
for pt in thresholds_key:
    row = {"Threshold": pt}
    for mname, mdata in models_for_dca.items():
        y_p_m    = mdata["y_prob"]
        y_b_m    = label_binarize(mdata["y_true"], classes=list(range(N_CLASSES)))
        nb_total = 0.0
        for c in range(N_CLASSES):
            pred_pos = y_p_m[:, c] >= pt
            tp = float(np.sum(pred_pos & (y_b_m[:, c] == 1)))
            fp = float(np.sum(pred_pos & (y_b_m[:, c] == 0)))
            nb_total += tp / n_samp - fp / n_samp * (pt / (1.0 - pt))
        row[mname] = round(nb_total / N_CLASSES, 4)
    row["Treat-All"]  = round(prev_macro - (1.0 - prev_macro) * pt / (1.0 - pt), 4)
    row["Treat-None"] = 0.0
    dca_nb_rows.append(row)

dca_nb_path = os.path.join(OUT_DIR, "dca_net_benefit_table.csv")
pd.DataFrame(dca_nb_rows).to_csv(dca_nb_path, index=False)
print(f"  Net benefit table    → {dca_nb_path}")
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
