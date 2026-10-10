"""Plot Decision Curve Analysis (DCA) Net Benefit from dca_combined.csv.

Generates a publication-quality figure comparing the net benefit of multiple
models (ResNet-50, Proposed, DivGI, Balanced-MixUp) across a range of
threshold probabilities, alongside the "Treat All" and "Treat None" reference
strategies.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_STYLES = {
    "Proposed":       {"color": "#D7263D", "marker": "o", "lw": 2.4, "ls": "-",  "zorder": 5},
    "ResNet-50":      {"color": "#1B6FB0", "marker": "s", "lw": 2.0, "ls": "-",  "zorder": 4},
    "DivGI":          {"color": "#2BA84A", "marker": "^", "lw": 2.0, "ls": "-",  "zorder": 3},
    "Balanced-MixUp": {"color": "#7B5EA7", "marker": "D", "lw": 2.0, "ls": "-",  "zorder": 2},
}


def compute_reference_curves(thresholds: np.ndarray, prevalence: float):
    """Compute 'Treat All' and 'Treat None' reference net-benefit curves."""
    treat_none = np.zeros_like(thresholds)
    treat_all = prevalence - (1.0 - prevalence) * (thresholds / (1.0 - thresholds))
    return treat_all, treat_none


def plot_dca(csv_path: str, output_path: str, prevalence: float | None = None,
             show_reference: bool = False, dpi: int = 300) -> None:
    df = pd.read_csv(csv_path)
    thresholds = df["Threshold"].to_numpy()
    model_cols = [c for c in df.columns if c != "Threshold"]
    values = df[model_cols].to_numpy()

    if prevalence is None:
        prevalence = float(df.loc[df["Threshold"].idxmin(), model_cols].max())

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    if show_reference:
        treat_all, treat_none = compute_reference_curves(thresholds, prevalence)
        ax.plot(thresholds, treat_all, color="#444444", ls="--", lw=1.5,
                label="Treat All", zorder=1)
        ax.plot(thresholds, treat_none, color="#888888", ls=":", lw=1.5,
                label="Treat None", zorder=1)

    for col in model_cols:
        style = MODEL_STYLES.get(col, {"color": None, "marker": "o",
                                       "lw": 2.0, "ls": "-", "zorder": 3})
        ax.plot(thresholds, df[col].to_numpy(),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.9,
                label=col,
                zorder=style["zorder"])

    ax.set_xlabel("Threshold Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Net Benefit", fontsize=12, fontweight="bold")
    ax.set_title("Decision Curve Analysis", fontsize=13, fontweight="bold", pad=10)

    ax.set_xticks(thresholds)
    ax.set_xlim(thresholds.min() - 0.02, thresholds.max() + 0.02)

    data_min = float(values.min())
    data_max = float(values.max())
    if show_reference:
        treat_all, _ = compute_reference_curves(thresholds, prevalence)
        data_min = min(data_min, float(treat_all.min()), 0.0)
        data_max = max(data_max, float(treat_all.max()))
    pad = (data_max - data_min) * 0.12 + 1e-4
    ax.set_ylim(data_min - pad, data_max + pad)

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

    leg = ax.legend(loc="best", frameon=True, fontsize=10,
                    edgecolor="#cccccc", facecolor="white", framealpha=0.95)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/figure3_dca/dca_combined.csv",
                        help="Path to dca_combined.csv")
    parser.add_argument("--out", default="results/figure3_dca/dca_net_benefit.png",
                        help="Output figure path (PNG)")
    parser.add_argument("--prevalence", type=float, default=None,
                        help="Disease prevalence for 'Treat All' reference "
                             "curve. Defaults to the max net benefit at the "
                             "lowest threshold.")
    parser.add_argument("--reference", action="store_true",
                        help="Overlay 'Treat All' / 'Treat None' reference "
                             "curves (off by default because the typical net "
                             "benefit range is narrow).")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_dca(
        csv_path=args.csv,
        output_path=args.out,
        prevalence=args.prevalence,
        show_reference=args.reference,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
