"""
delong_pairwise.py
==================
Compute all pairwise DeLong p-values for the macro-averaged OvR AUC table
(ResNet-50, Proposed/LAG, DivGI, Balanced-MixUp)  —  paper Tables 2 and 3.

Reads results/per_model/<model>/scores.npz (written by statistics/evaluate_cgle.py
and baselines/*/evaluate_stats.py) and writes
results/table2_3_delong/delong_pairwise.csv.

    python statistics/delong_pairwise.py

Notes on method
---------------
* ResNet-50, DivGI, and Balanced-MixUp share the SAME test-sample ordering
  → paired DeLong test (accounts for inter-classifier correlation).
  Method: per-class paired z-statistics averaged across 15 OvR classes.

* LAG (Proposed) was evaluated with a different DataLoader iteration order
  AND uses a different class-label numbering (checkpoint order vs alphabetical).
  → independent (non-paired) DeLong z-test, which ignores the between-model
  correlation and is therefore conservative but valid.
  Method: per-class independent z = (AUC_a − AUC_b) / sqrt(Var_a + Var_b),
  averaged across 15 OvR classes.

Reference: DeLong et al. (1988) Biometrics 44:837-845.
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import label_binarize

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from stat_utils import _placement_values, macro_delong_comparison

# ── class definitions ─────────────────────────────────────────────────────────
# Alphabetical order used by ImageFolder (ResNet-50, DivGI, Balanced-MixUp)
ALPH_CLASSES = [
    "bbps-0-1", "bbps-2-3", "cecum", "dyed-lifted-polyps",
    "dyed-resection-margins", "esophagitis-a", "esophagitis-b-d",
    "impacted-stool", "polyps", "pylorus", "retroflex-rectum",
    "retroflex-stomach", "ulcerative-colitis-mild", "ulcerative-colitis-severe",
    "z-line",
]
N_CLASSES = len(ALPH_CLASSES)

# LAG checkpoint class order (from delong_per_class_auc.csv)
LAG_CLASSES = [
    "cecum", "esophagitis-b-d", "retroflex-rectum", "retroflex-stomach",
    "bbps-2-3", "ulcerative-colitis-mild", "bbps-0-1",
    "dyed-resection-margins", "pylorus", "polyps", "esophagitis-a",
    "dyed-lifted-polyps", "z-line", "impacted-stool",
    "ulcerative-colitis-severe",
]
# Permutation: lag_idx -> alph_idx
LAG_TO_ALPH = [ALPH_CLASSES.index(c) for c in LAG_CLASSES]

# ── load scores ───────────────────────────────────────────────────────────────
_PER_MODEL = os.path.join(_ROOT, "results", "per_model")
SCORE_PATHS = {
    "ResNet-50":       os.path.join(_PER_MODEL, "resnet50",       "scores.npz"),
    "Proposed":        os.path.join(_PER_MODEL, "cgle",           "scores.npz"),
    "DivGI":           os.path.join(_PER_MODEL, "divgi",          "scores.npz"),
    "Balanced-MixUp":  os.path.join(_PER_MODEL, "balanced_mixup", "scores.npz"),
}

raw = {}
for name, path in SCORE_PATHS.items():
    d = np.load(path)
    raw[name] = {"y_true": d["y_true"], "y_prob": d["y_prob"].astype(np.float64)}
    print(f"Loaded {name}: {d['y_true'].shape[0]} samples")

# Re-map LAG to alphabetical class ordering so probabilities align with ALPH_CLASSES
lag_prob_alph = np.empty_like(raw["Proposed"]["y_prob"])
for lag_idx, alph_idx in enumerate(LAG_TO_ALPH):
    lag_prob_alph[:, alph_idx] = raw["Proposed"]["y_prob"][:, lag_idx]
lag_y_true_alph = np.array([LAG_TO_ALPH[c] for c in raw["Proposed"]["y_true"]])

# Unified data dict: all y_true / y_prob in alphabetical class space
data = {
    "ResNet-50":      {"y_true": raw["ResNet-50"]["y_true"],       "y_prob": raw["ResNet-50"]["y_prob"]},
    "Proposed":       {"y_true": lag_y_true_alph,                  "y_prob": lag_prob_alph},
    "DivGI":          {"y_true": raw["DivGI"]["y_true"],           "y_prob": raw["DivGI"]["y_prob"]},
    "Balanced-MixUp": {"y_true": raw["Balanced-MixUp"]["y_true"],  "y_prob": raw["Balanced-MixUp"]["y_prob"]},
}


# ── DeLong helpers ────────────────────────────────────────────────────────────

def delong_auc_variance(y_true_bin, y_score):
    """DeLong variance of AUC for a single binary classifier."""
    pos = y_score[y_true_bin == 1]
    neg = y_score[y_true_bin == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan
    V10, V01 = _placement_values(y_true_bin, y_score)
    auc = V10.mean()
    var = np.var(V10, ddof=1) / len(pos) + np.var(V01, ddof=1) / len(neg)
    return auc, var


def independent_macro_delong(y_true_a, y_prob_a, y_true_b, y_prob_b):
    """
    Non-paired macro DeLong test (conservative) for classifiers evaluated on
    potentially differently-ordered but equivalent test sets.

    Uses z = (AUC_a - AUC_b) / sqrt(Var_a + Var_b) per class, then averages z.
    """
    y_bin_a = label_binarize(y_true_a, classes=list(range(N_CLASSES)))
    y_bin_b = label_binarize(y_true_b, classes=list(range(N_CLASSES)))
    aucs_a, aucs_b, zs = [], [], []
    for i in range(N_CLASSES):
        auc_a, var_a = delong_auc_variance(y_bin_a[:, i], y_prob_a[:, i])
        auc_b, var_b = delong_auc_variance(y_bin_b[:, i], y_prob_b[:, i])
        if np.isnan(var_a) or np.isnan(var_b):
            continue
        total_var = var_a + var_b
        if total_var <= 0:
            continue
        zs.append((auc_a - auc_b) / np.sqrt(total_var))
        aucs_a.append(auc_a)
        aucs_b.append(auc_b)
    if not zs:
        return np.nan, np.nan, np.nan, np.nan
    z_macro = float(np.mean(zs))
    p_macro  = float(2.0 * stats.norm.sf(abs(z_macro)))
    return float(np.mean(aucs_a)), float(np.mean(aucs_b)), z_macro, p_macro


# ── pairwise comparisons ──────────────────────────────────────────────────────

# Models that share the same test-sample ordering (alphabetical DataLoader)
ALIGNED = {"ResNet-50", "DivGI", "Balanced-MixUp"}

PAIRS = [
    ("ResNet-50",      "Proposed"),
    ("ResNet-50",      "DivGI"),
    ("ResNet-50",      "Balanced-MixUp"),
    ("Proposed",       "DivGI"),
    ("Proposed",       "Balanced-MixUp"),
    ("DivGI",          "Balanced-MixUp"),
]

print("\n" + "=" * 78)
print("  Pairwise DeLong Test — Macro OvR AUC (average-z, 15 classes)")
print("=" * 78)

results = {}
for model_a, model_b in PAIRS:
    paired = model_a in ALIGNED and model_b in ALIGNED
    da, db  = data[model_a], data[model_b]

    if paired:
        # Paired test: both models on exactly the same ordered test samples
        auc_a, auc_b, z, p = macro_delong_comparison(
            da["y_true"], da["y_prob"], db["y_prob"], N_CLASSES, ALPH_CLASSES)
        method = "paired"
    else:
        # Independent test: sample ordering differs (conservative)
        auc_a, auc_b, z, p = independent_macro_delong(
            da["y_true"], da["y_prob"], db["y_true"], db["y_prob"])
        method = "independent"

    key = f"{model_a} vs {model_b}"
    results[key] = p
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
    print(f"  {key:<40}  AUC={auc_a:.4f} vs {auc_b:.4f}  "
          f"z={z:+.3f}  p={p:.4e}  [{sig}] ({method})")

print("=" * 78)
print("  * p<0.05   ** p<0.01   ns = not significant")
print("  paired   = same sample ordering; independent = conservative (LAG ordering differs)")

# ── save pairwise p-values ────────────────────────────────────────────────────
OUT_CSV = os.path.join(_ROOT, "results", "table2_3_delong", "delong_pairwise.csv")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pd.DataFrame([{"Model Pair": k, "p-value": v} for k, v in results.items()]).to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
