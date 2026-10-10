"""
ablation_utils.py
=================
Shared statistical utilities for ablation-study inference scripts.

Provides:
  - DeLong (1988) AUC variance → 95% CI for a single model
  - DeLong paired test         → significance test between two models (same test set)
  - Bootstrap 95% CI           → Accuracy, Sensitivity, Specificity, F1, macro-AUC
  - Per-class Sensitivity & Specificity from confusion matrix
  - Calibration analysis       → reliability diagram, ECE, Brier score
  - Decision Curve Analysis    → net benefit vs threshold probability
  - ROC curve plot             → per-class + macro average
  - Summary print & CSV save
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, roc_curve, confusion_matrix,
)
from sklearn.preprocessing import label_binarize


# ══════════════════════════════════════════════════════════════════
#  DeLong (1988) — AUC CI for a single model
# ══════════════════════════════════════════════════════════════════

def _placement_values(y_true_bin, y_score):
    """
    Compute the structural (placement) values needed for DeLong variance.

    V10[i]  = P( score_positive_i > score_negative )   for each positive
    V01[j]  = P( score_positive   > score_negative_j ) for each negative
    """
    pos = y_score[y_true_bin == 1]
    neg = y_score[y_true_bin == 0]
    V10 = np.array(
        [(np.sum(p > neg) + 0.5 * np.sum(p == neg)) / len(neg) for p in pos]
    )
    V01 = np.array(
        [(np.sum(pos > n) + 0.5 * np.sum(pos == n)) / len(pos) for n in neg]
    )
    return V10, V01


def delong_auc_ci(y_true_bin, y_score, alpha=0.05):
    """
    DeLong (1988) variance-based confidence interval for a single binary AUC.

    Parameters
    ----------
    y_true_bin : array-like, shape (n,)  — binary ground truth (0/1)
    y_score    : array-like, shape (n,)  — predicted probability for the positive class
    alpha      : float                   — significance level (default 0.05 → 95 % CI)

    Returns
    -------
    auc      : float
    ci_low   : float
    ci_high  : float
    """
    y_true_bin = np.asarray(y_true_bin)
    y_score    = np.asarray(y_score)
    pos = y_score[y_true_bin == 1]
    neg = y_score[y_true_bin == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan, np.nan

    V10, V01 = _placement_values(y_true_bin, y_score)
    auc      = V10.mean()
    var_auc  = np.var(V10, ddof=1) / len(pos) + np.var(V01, ddof=1) / len(neg)
    se       = np.sqrt(var_auc)
    z        = stats.norm.ppf(1.0 - alpha / 2.0)
    return (
        float(auc),
        float(np.clip(auc - z * se, 0.0, 1.0)),
        float(np.clip(auc + z * se, 0.0, 1.0)),
    )


# ══════════════════════════════════════════════════════════════════
#  DeLong paired significance test — two models, same test set
# ══════════════════════════════════════════════════════════════════

def delong_paired_test(y_true_bin, scores_a, scores_b):
    """
    DeLong (1988) correlated-AUC significance test (two-sided).

    Compares AUC of model A vs model B evaluated on the same test set.
    Uses the 2×2 covariance structure to account for correlation.

    Parameters
    ----------
    y_true_bin : array-like (n,)  — binary ground truth
    scores_a   : array-like (n,)  — model A predicted probability
    scores_b   : array-like (n,)  — model B predicted probability

    Returns
    -------
    auc_a  : float
    auc_b  : float
    z_stat : float
    p_val  : float   (two-sided)
    """
    y_true_bin = np.asarray(y_true_bin)
    n_pos = int((y_true_bin == 1).sum())
    n_neg = int((y_true_bin == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan, np.nan

    V10_a, V01_a = _placement_values(y_true_bin, np.asarray(scores_a))
    V10_b, V01_b = _placement_values(y_true_bin, np.asarray(scores_b))

    auc_a = V10_a.mean()
    auc_b = V10_b.mean()

    # 2×2 covariance matrices for positive / negative structural components
    S10 = np.cov(np.stack([V10_a, V10_b]), ddof=1) / n_pos
    S01 = np.cov(np.stack([V01_a, V01_b]), ddof=1) / n_neg
    S   = S10 + S01        # covariance of [AUC_a, AUC_b]

    var_diff = S[0, 0] + S[1, 1] - 2.0 * S[0, 1]
    if var_diff <= 0.0:
        return auc_a, auc_b, np.nan, np.nan

    z     = (auc_a - auc_b) / np.sqrt(var_diff)
    p_val = 2.0 * stats.norm.sf(abs(z))
    return float(auc_a), float(auc_b), float(z), float(p_val)


# ══════════════════════════════════════════════════════════════════
#  Macro OvR AUC with per-class DeLong CI
# ══════════════════════════════════════════════════════════════════

def macro_auc_with_delong_ci(y_true, y_prob, n_classes, class_names, alpha=0.05):
    """
    Compute per-class OvR AUC + DeLong 95 % CI, then macro-average.

    Returns
    -------
    macro_auc : float
    per_class : dict  {class_name: {"auc": …, "ci_low": …, "ci_high": …}}
    """
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))
    per_class = {}
    valid_aucs = []

    for i, cls in enumerate(class_names):
        auc, ci_low, ci_high = delong_auc_ci(y_bin[:, i], y_prob[:, i], alpha=alpha)
        per_class[cls] = {"auc": auc, "ci_low": ci_low, "ci_high": ci_high}
        if not np.isnan(auc):
            valid_aucs.append(auc)

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else np.nan
    return macro_auc, per_class


# ══════════════════════════════════════════════════════════════════
#  Bootstrap 95 % CI — all scalar metrics
# ══════════════════════════════════════════════════════════════════

def bootstrap_ci(y_true, y_pred, y_prob, n_classes,
                 n_boot=1000, alpha=0.05, seed=42):
    """
    Non-parametric bootstrap 95 % CI for:
      Accuracy, Sensitivity (macro recall), Specificity (macro),
      F1 (macro), Macro-AUC.

    Returns
    -------
    dict  {metric_name: (ci_low, ci_high)}
    """
    rng       = np.random.default_rng(seed)
    y_true    = np.asarray(y_true)
    y_pred    = np.asarray(y_pred)
    y_prob    = np.asarray(y_prob)
    n_samples = len(y_true)
    labels    = list(range(n_classes))

    boot = {"acc": [], "sens": [], "spec": [], "f1": [], "auc": []}

    for _ in range(n_boot):
        idx  = rng.integers(0, n_samples, size=n_samples)
        yt   = y_true[idx]
        yp   = y_pred[idx]
        ypr  = y_prob[idx]

        boot["acc"].append(accuracy_score(yt, yp))
        boot["sens"].append(recall_score(yt, yp, average="macro", zero_division=0))

        cm_ = confusion_matrix(yt, yp, labels=labels)
        spec_cls = []
        for c in range(n_classes):
            tp  = cm_[c, c]
            fn  = cm_[c, :].sum() - tp
            fp  = cm_[:, c].sum() - tp
            tn  = cm_.sum() - tp - fn - fp
            spec_cls.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        boot["spec"].append(float(np.mean(spec_cls)))

        boot["f1"].append(f1_score(yt, yp, average="macro", zero_division=0))

        try:
            yb = label_binarize(yt, classes=labels)
            if yb.shape[1] == n_classes:
                boot["auc"].append(
                    roc_auc_score(yb, ypr, average="macro", multi_class="ovr")
                )
        except Exception:
            pass

    lo, hi = alpha / 2.0 * 100, (1.0 - alpha / 2.0) * 100
    return {
        k: (float(np.percentile(v, lo)), float(np.percentile(v, hi)))
        if v else (np.nan, np.nan)
        for k, v in boot.items()
    }


# ══════════════════════════════════════════════════════════════════
#  Per-class Sensitivity & Specificity
# ══════════════════════════════════════════════════════════════════

def per_class_sens_spec(y_true, y_pred, n_classes):
    """
    Derive per-class sensitivity (TPR) and specificity (TNR) from
    the confusion matrix.

    Returns
    -------
    sens : np.ndarray (n_classes,)
    spec : np.ndarray (n_classes,)
    """
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    sens = np.zeros(n_classes)
    spec = np.zeros(n_classes)
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec[c] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, spec


# ══════════════════════════════════════════════════════════════════
#  Calibration Analysis
# ══════════════════════════════════════════════════════════════════

def calibration_analysis(y_true, y_prob, class_names, out_dir, variant_name,
                         n_bins=10):
    """
    Produce per-class reliability (calibration) diagrams.
    Compute Expected Calibration Error (ECE) and Brier score.

    Saves  : <out_dir>/calibration_reliability.png
    Returns: dict with macro_ece, macro_brier, per_class_brier
    """
    os.makedirs(out_dir, exist_ok=True)
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    ncols = min(4, n_classes)
    nrows = (n_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes = np.array(axes).flatten()

    brier_scores = []
    ece_scores   = []

    for i, cls in enumerate(class_names):
        prob_i    = y_prob[:, i]
        label_i   = y_bin[:, i]
        brier_scores.append(float(brier_score_loss(label_i, prob_i)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frac_pos, mean_pred = calibration_curve(
                label_i, prob_i, n_bins=n_bins, strategy="uniform"
            )

        # Weighted ECE: weight each bin by its fraction of total samples
        bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
        bin_counts = np.histogram(prob_i, bins=bin_edges)[0]
        total      = bin_counts.sum()
        bin_w      = bin_counts / total if total > 0 else np.ones(n_bins) / n_bins
        # calibration_curve may return fewer bins; align lengths
        n_valid = len(frac_pos)
        ece_scores.append(float(np.sum(np.abs(frac_pos - mean_pred) * bin_w[:n_valid])))

        ax = axes[i]
        ax.plot(mean_pred, frac_pos, "s-", color="royalblue", lw=1.5,
                ms=4, label="Model")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect")
        ax.set_title(cls, fontsize=8, pad=3)
        ax.set_xlabel("Mean predicted prob.", fontsize=7)
        ax.set_ylabel("Fraction of positives", fontsize=7)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper left")
        ax.text(0.98, 0.04,
                f"Brier={brier_scores[-1]:.3f}\nECE={ece_scores[-1]:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Calibration Reliability Diagrams — {variant_name}",
                 fontsize=11, y=1.005)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "calibration_reliability.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    macro_ece   = float(np.mean(ece_scores))
    macro_brier = float(np.mean(brier_scores))
    return {
        "macro_ece":       macro_ece,
        "macro_brier":     macro_brier,
        "per_class_brier": dict(zip(class_names, brier_scores)),
        "per_class_ece":   dict(zip(class_names, ece_scores)),
        "fig_path":        fig_path,
    }


# ══════════════════════════════════════════════════════════════════
#  Decision Curve Analysis (DCA)
# ══════════════════════════════════════════════════════════════════

def decision_curve_analysis(y_true, y_prob, class_names, out_dir, variant_name,
                            thresholds=None):
    """
    Per-class (OvR) Decision Curve Analysis.

    Net Benefit = TP/N  –  FP/N × pt/(1–pt)

    Saves  : <out_dir>/dca.png
    Returns: dict  {class_name: {"thresholds": …, "net_benefit": …}}
    """
    os.makedirs(out_dir, exist_ok=True)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 199)

    n_classes = len(class_names)
    n_samples  = len(y_true)
    y_bin      = label_binarize(y_true, classes=list(range(n_classes)))

    ncols = min(4, n_classes)
    nrows = (n_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes = np.array(axes).flatten()

    dca_out = {}

    for i, cls in enumerate(class_names):
        y_bin_cls  = y_bin[:, i].astype(float)
        prevalence = y_bin_cls.mean()
        prob_cls   = y_prob[:, i]

        net_benefit = np.empty(len(thresholds))
        treat_all   = np.empty(len(thresholds))

        for k, pt in enumerate(thresholds):
            predicted_pos = prob_cls >= pt
            tp = float(np.sum(predicted_pos & (y_bin_cls == 1)))
            fp = float(np.sum(predicted_pos & (y_bin_cls == 0)))
            net_benefit[k] = tp / n_samples - fp / n_samples * (pt / (1.0 - pt))
            treat_all[k]   = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))

        dca_out[cls] = {"thresholds": thresholds, "net_benefit": net_benefit}

        ax = axes[i]
        ax.plot(thresholds, net_benefit, color="royalblue", lw=2.0,
                label="Model")
        ax.plot(thresholds, treat_all,   color="tomato",    lw=1.5,
                ls="--", label="Treat All")
        ax.axhline(0, color="gray", lw=1.0, ls=":", label="Treat None")
        ax.set_title(cls, fontsize=8, pad=3)
        ax.set_xlabel("Threshold probability", fontsize=7)
        ax.set_ylabel("Net benefit",           fontsize=7)
        ax.set_xlim([0, 1])
        y_max = max(prevalence + 0.05, 0.15)
        ax.set_ylim([-0.05, y_max])
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="upper right")

    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Decision Curve Analysis — {variant_name}",
                 fontsize=11, y=1.005)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "dca.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"dca": dca_out, "fig_path": fig_path}


# ══════════════════════════════════════════════════════════════════
#  ROC Curves (per-class + macro average)
# ══════════════════════════════════════════════════════════════════

def plot_roc_curves(y_true, y_prob, class_names, out_dir, variant_name):
    """
    Overlay per-class ROC curves (thin) plus macro-average (thick).
    Saves  : <out_dir>/roc_curves.png
    """
    os.makedirs(out_dir, exist_ok=True)
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))
    mean_fpr  = np.linspace(0, 1, 300)
    tprs, aucs = [], []

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.get_cmap("tab20", n_classes)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_val      = float(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(auc_val)
        ax.plot(fpr, tpr, alpha=0.25, lw=0.8, color=cmap(i))

    mean_tpr      = np.mean(tprs, axis=0)
    mean_tpr[-1]  = 1.0
    mean_auc      = float(np.mean(aucs))
    std_auc       = float(np.std(aucs))

    ax.plot(mean_fpr, mean_tpr, color="royalblue", lw=2.5,
            label=f"Macro-avg ROC  AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curves — {variant_name}", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


# ══════════════════════════════════════════════════════════════════
#  Confusion-matrix plot
# ══════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, class_names, out_dir, variant_name):
    """Save a seaborn confusion-matrix heat-map."""
    import seaborn as sns
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={"size": 8})
    ax.set_title(f"Confusion Matrix — {variant_name}", fontsize=14, pad=12)
    ax.set_xlabel("Predicted label", fontsize=11, labelpad=10)
    ax.set_ylabel("True label",      fontsize=11, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  ha="right",  fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


# ══════════════════════════════════════════════════════════════════
#  DeLong significance test vs reference model (multi-class, macro)
# ══════════════════════════════════════════════════════════════════

def delong_vs_reference(y_true, y_prob_ablation, y_prob_ref, n_classes,
                         class_names, alpha=0.05):
    """
    Run per-class DeLong paired test (ablation vs full model) and
    apply Bonferroni correction for multiple comparisons.

    Returns
    -------
    results : dict
        {class_name: {auc_ablation, auc_ref, z, p_raw, p_bonf, significant}}
    macro_summary : dict
        {mean_z, mean_p_raw, n_significant}
    """
    y_bin    = label_binarize(y_true, classes=list(range(n_classes)))
    results  = {}
    raw_pvals = []

    for i, cls in enumerate(class_names):
        auc_a, auc_b, z, p = delong_paired_test(
            y_bin[:, i], y_prob_ablation[:, i], y_prob_ref[:, i]
        )
        results[cls] = {
            "auc_ablation": auc_a,
            "auc_reference": auc_b,
            "z_stat": z,
            "p_raw":  p,
        }
        if not np.isnan(p):
            raw_pvals.append(p)

    # Bonferroni correction
    n_tests = len(raw_pvals)
    for cls in class_names:
        p_raw = results[cls]["p_raw"]
        p_bonf = min(float(p_raw) * n_tests, 1.0) if not np.isnan(p_raw) else np.nan
        results[cls]["p_bonferroni"] = p_bonf
        results[cls]["significant"]  = (p_bonf < alpha) if not np.isnan(p_bonf) else False

    n_sig = sum(1 for v in results.values() if v.get("significant", False))
    return results, {"n_classes_tested": n_tests, "n_significant_bonf": n_sig}


# ══════════════════════════════════════════════════════════════════
#  Summary print & CSV save
# ══════════════════════════════════════════════════════════════════

def print_summary(metrics: dict, variant_name: str):
    """Pretty-print the scalar metrics table."""
    w = 36
    print(f"\n{'═' * (w + 26)}")
    print(f"  {variant_name}  —  Test-set Evaluation")
    print(f"{'═' * (w + 26)}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<{w}}: {v:.4f}")
        elif isinstance(v, tuple) and len(v) == 2:
            print(f"  {k:<{w}}: ({v[0]:.4f},  {v[1]:.4f})")
        else:
            print(f"  {k:<{w}}: {v}")
    print(f"{'═' * (w + 26)}\n")


def save_metrics_csv(metrics: dict, out_dir: str, filename: str = "metrics.csv"):
    """Flatten the metrics dict (tuples → _low/_high) and save to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, float) or isinstance(v, (int, np.floating)):
            flat[k] = float(v)
        elif isinstance(v, tuple) and len(v) == 2:
            flat[f"{k}_CI95_low"]  = float(v[0])
            flat[f"{k}_CI95_high"] = float(v[1])
    csv_path = os.path.join(out_dir, filename)
    pd.DataFrame([flat]).to_csv(csv_path, index=False)
    print(f"  Metrics CSV  → {csv_path}")
    return csv_path


def save_delong_vs_ref_csv(delong_results: dict, out_dir: str,
                            filename: str = "delong_vs_reference.csv"):
    """Save per-class DeLong comparison results to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for cls, vals in delong_results.items():
        row = {"class": cls}
        row.update(vals)
        rows.append(row)
    csv_path = os.path.join(out_dir, filename)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  DeLong CSV   → {csv_path}")
    return csv_path


# ══════════════════════════════════════════════════════════════════
#  Calibration Slope & Intercept  (per-class logistic fit)
# ══════════════════════════════════════════════════════════════════

def calibration_slope_intercept(y_true_bin, y_prob_cls, eps=1e-7):
    """
    Logistic calibration fit:  y ~ logistic(intercept + slope * logit(p))

    Calibration slope  → ideal = 1  (< 1: over-confident, > 1: under-confident)
    Calibration intercept → ideal = 0  (calibration-in-the-large)

    Returns (slope, intercept) or (nan, nan) if fit fails.
    """
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y_true_bin, dtype=int)
    p = np.clip(y_prob_cls, eps, 1.0 - eps)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs",
                                max_iter=1000, random_state=42)
        lr.fit(logit_p, y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan


# ══════════════════════════════════════════════════════════════════
#  Hosmer–Lemeshow goodness-of-fit test
# ══════════════════════════════════════════════════════════════════

def hosmer_lemeshow_test(y_true_bin, y_prob_cls, n_bins=10):
    """
    Hosmer–Lemeshow chi-squared test (decile-of-risk grouping).

    High p-value (> 0.05) → model is well-calibrated.

    Returns (chi2_stat, p_value).
    """
    from scipy.stats import chi2 as chi2_dist
    y = np.asarray(y_true_bin, dtype=float)
    p = np.asarray(y_prob_cls, dtype=float)
    order    = np.argsort(p)
    y_s      = y[order]
    p_s      = p[order]
    bins     = np.array_split(np.arange(len(y)), n_bins)
    hl_stat  = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        obs_pos = y_s[b].sum()
        exp_pos = p_s[b].sum()
        obs_neg = len(b) - obs_pos
        exp_neg = len(b) - exp_pos
        if exp_pos > 0:
            hl_stat += (obs_pos - exp_pos) ** 2 / exp_pos
        if exp_neg > 0:
            hl_stat += (obs_neg - exp_neg) ** 2 / exp_neg
    df    = max(n_bins - 2, 1)
    p_val = float(1.0 - chi2_dist.cdf(hl_stat, df))
    return float(hl_stat), p_val


# ══════════════════════════════════════════════════════════════════
#  Macro calibration stats  (aggregates per-class fits)
# ══════════════════════════════════════════════════════════════════

def macro_calibration_stats(y_true, y_prob, n_classes, class_names, n_bins=10):
    """
    Per-class then macro-averaged calibration slope, intercept, and HL test.

    Returns
    -------
    dict with:
        per_class       : {cls: {slope, intercept, hl_chi2, hl_p}}
        macro_slope     : float
        macro_intercept : float
        macro_hl_p      : float  (mean of per-class HL p-values)
    """
    y_bin      = label_binarize(y_true, classes=list(range(n_classes)))
    per_class  = {}
    slopes, intercepts, hl_ps = [], [], []

    for i, cls in enumerate(class_names):
        slope, intercept = calibration_slope_intercept(y_bin[:, i], y_prob[:, i])
        hl_chi2, hl_p    = hosmer_lemeshow_test(y_bin[:, i], y_prob[:, i], n_bins)
        per_class[cls]   = {"slope": slope, "intercept": intercept,
                            "hl_chi2": hl_chi2, "hl_p": hl_p}
        if not np.isnan(slope):      slopes.append(slope)
        if not np.isnan(intercept):  intercepts.append(intercept)
        if not np.isnan(hl_p):       hl_ps.append(hl_p)

    return {
        "per_class":        per_class,
        "macro_slope":      float(np.mean(slopes))      if slopes      else np.nan,
        "macro_intercept":  float(np.mean(intercepts))  if intercepts  else np.nan,
        "macro_hl_p":       float(np.mean(hl_ps))       if hl_ps       else np.nan,
    }


# ══════════════════════════════════════════════════════════════════
#  Macro DeLong comparison  (two models, average-z approach)
# ══════════════════════════════════════════════════════════════════

def macro_delong_comparison(y_true, y_prob_a, y_prob_b, n_classes, class_names):
    """
    Per-class DeLong paired test → macro summary via average-z statistic.

    Returns (auc_a_macro, auc_b_macro, z_macro, p_macro).
    p_macro from two-sided normal test on mean(z_i).
    """
    y_bin  = label_binarize(y_true, classes=list(range(n_classes)))
    aucs_a, aucs_b, zs = [], [], []
    for i in range(n_classes):
        a, b, z, _ = delong_paired_test(
            y_bin[:, i], y_prob_a[:, i], y_prob_b[:, i]
        )
        if not np.isnan(z):
            aucs_a.append(a)
            aucs_b.append(b)
            zs.append(z)
    if not zs:
        return np.nan, np.nan, np.nan, np.nan
    z_macro = float(np.mean(zs))
    p_macro = float(2.0 * stats.norm.sf(abs(z_macro)))
    return float(np.mean(aucs_a)), float(np.mean(aucs_b)), z_macro, p_macro


# ══════════════════════════════════════════════════════════════════
#  Combined macro DCA  (overlay multiple models on one plot)
# ══════════════════════════════════════════════════════════════════

def plot_combined_dca_macro(models_data, out_dir,
                            title="Decision Curve Analysis",
                            thresholds=None):
    """
    Overlay macro-averaged net benefit curves for multiple models.

    Parameters
    ----------
    models_data : dict  {model_name: {"y_true": …, "y_prob": …, "n_classes": …}}
    out_dir     : str
    title       : str

    Saves : <out_dir>/dca_combined_macro.png
    """
    os.makedirs(out_dir, exist_ok=True)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.80, 200)

    colors = ["royalblue", "tomato", "seagreen", "darkorange", "purple", "brown"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (mname, mdata) in enumerate(models_data.items()):
        y_true_m  = mdata["y_true"]
        y_prob_m  = mdata["y_prob"]
        n_cls     = mdata["n_classes"]
        y_bin_m   = label_binarize(y_true_m, classes=list(range(n_cls)))
        n_samp    = len(y_true_m)
        macro_nb  = np.zeros(len(thresholds))

        for c in range(n_cls):
            for k, pt in enumerate(thresholds):
                pred_pos = y_prob_m[:, c] >= pt
                tp = float(np.sum(pred_pos & (y_bin_m[:, c] == 1)))
                fp = float(np.sum(pred_pos & (y_bin_m[:, c] == 0)))
                macro_nb[k] += tp / n_samp - fp / n_samp * (pt / (1.0 - pt))
        macro_nb /= n_cls

        c_col = colors[idx % len(colors)]
        lw    = 2.5 if idx == 0 else 1.8
        ax.plot(thresholds, macro_nb, color=c_col, lw=lw, label=mname)

    # Treat-all / treat-none from first model
    first       = next(iter(models_data.values()))
    y_bin_first = label_binarize(first["y_true"], classes=list(range(first["n_classes"])))
    prevalence  = float(y_bin_first.mean(axis=0).mean())
    treat_all   = prevalence - (1.0 - prevalence) * (thresholds / (1.0 - thresholds))
    ax.plot(thresholds, treat_all, "k--", lw=1.2, label="Treat All")
    ax.axhline(0, color="gray", lw=1.0, ls=":", label="Treat None")

    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit (macro avg.)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim([0.0, 0.80])
    ax.set_ylim([-0.05, prevalence + 0.10])
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "dca_combined_macro.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


# ══════════════════════════════════════════════════════════════════
#  Paper-ready table savers
# ══════════════════════════════════════════════════════════════════

def save_paper_discrimination_table(rows, out_dir,
                                    filename="paper_table_discrimination.csv"):
    """
    Save discrimination table in paper-ready format.
    Expected columns: Model | AUC | 95% CI (DeLong) | p-value vs Reference
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Paper discrimination table → {path}")
    return path


def save_paper_calibration_table(rows, out_dir,
                                 filename="paper_table_calibration.csv"):
    """
    Save calibration table in paper-ready format.
    Expected columns: Model | Brier Score | Cal. Slope | Cal. Intercept | HL p-value
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Paper calibration table    → {path}")
    return path
