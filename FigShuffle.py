"""
Figure: Geometric Shuffle Control Results.

Generates figures showing that transformers encode belief structure beyond
next-token prediction, using the geometric shuffle control experiment.

Usage:
    uv run python FigShuffle.py --shuffle_dir shuffle_results --outdir figures/shuffle
"""
import argparse
import os
import glob
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path


def load_shuffle_results(shuffle_dir: str) -> dict:
    """Load all shuffle analysis results from a directory."""
    results = {}
    for fpath in sorted(glob.glob(os.path.join(shuffle_dir, "*_shuffle.joblib"))):
        data = joblib.load(fpath)
        run_name = os.path.basename(fpath).replace("_shuffle.joblib", "")
        results[run_name] = data
    return results


def get_run_label(config: dict) -> str:
    """Create a short label from process params."""
    pc = config["process_config"]
    phi = pc.get("phi", 0)
    theta = pc.get("theta", 0)
    eps = pc.get("epsilon", 0)
    return f"$\\varphi$={phi:.1f}\n$\\theta$={theta:.2f}\n$\\varepsilon$={eps:.2f}"


def fig_rmse_comparison(all_results: dict, outdir: Path):
    """Bar chart comparing original RMSE vs shuffled RMSE across experiments."""
    runs = sorted(all_results.keys())
    n = len(runs)

    fig, ax = plt.subplots(figsize=(max(10, 2 * n), 5))

    x = np.arange(n)
    width = 0.35

    orig_vals = []
    shuf_vals = []
    shuf_stds = []
    labels = []

    for run_name in runs:
        data = all_results[run_name]
        sc = data["results"]["combined"]
        orig_vals.append(sc["mse_original"])
        shuf_vals.append(sc["mse_shuffle_mean"])
        shuf_stds.append(sc["mse_shuffle_std"])
        labels.append(get_run_label(data["config"]))

    orig_vals = np.array(orig_vals)
    shuf_vals = np.array(shuf_vals)
    shuf_stds = np.array(shuf_stds)

    bars1 = ax.bar(x - width / 2, orig_vals, width, label="Original beliefs",
                   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, shuf_vals, width, label="Shuffled beliefs",
                   color="#F44336", alpha=0.85, yerr=shuf_stds, capsize=3)

    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel("Weighted RMSE", fontsize=11)
    ax.set_title("Belief State Regression: Original vs Shuffled Targets\n"
                 "(combined layers, final checkpoint)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(outdir / "shuffle_rmse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shuffle_rmse_comparison.png")


def fig_effect_sizes(all_results: dict, outdir: Path):
    """Bar chart of effect sizes with p-values annotated."""
    runs = sorted(all_results.keys())
    n = len(runs)

    fig, ax = plt.subplots(figsize=(max(10, 2 * n), 5))

    effects = []
    pvals = []
    labels = []
    fracs = []

    for run_name in runs:
        data = all_results[run_name]
        sc = data["results"]["combined"]
        effects.append(sc["effect_size"] * 100)  # as percentage
        pvals.append(sc["p_value"])
        fracs.append(sc["frac_beyond"] * 100)
        labels.append(get_run_label(data["config"]))

    x = np.arange(n)
    colors = ["#4CAF50" if p < 0.05 else "#FF9800" if p < 0.1 else "#9E9E9E"
              for p in pvals]

    bars = ax.bar(x, effects, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    # Annotate with p-values
    for i, (bar, p, frac) in enumerate(zip(bars, pvals, fracs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"p={p:.3f}\n{frac:.0f}% beyond",
                ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel("Effect size (%)", fontsize=11)
    ax.set_title("Shuffle Control Effect Size\n"
                 "(% increase in RMSE when beyond-next-token component is shuffled)",
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(y=0, color="black", linewidth=0.5)

    legend_elements = [
        Patch(facecolor="#4CAF50", label="p < 0.05 (significant)"),
        Patch(facecolor="#FF9800", label="p < 0.10 (marginal)"),
        Patch(facecolor="#9E9E9E", label="p >= 0.10 (not significant)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(outdir / "shuffle_effect_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shuffle_effect_sizes.png")


def fig_shuffle_distributions(all_results: dict, outdir: Path):
    """Histogram of shuffled RMSE distribution vs original for each experiment."""
    runs = sorted(all_results.keys())
    n = len(runs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, run_name in enumerate(runs):
        ax = axes[i]
        data = all_results[run_name]
        sc = data["results"]["combined"]

        shuffled = np.array(sc["mse_shuffled"])
        orig = sc["mse_original"]

        ax.hist(shuffled, bins=30, color="#F44336", alpha=0.7, edgecolor="black",
                linewidth=0.5, label="Shuffled")
        ax.axvline(orig, color="#2196F3", linewidth=2, linestyle="--",
                   label=f"Original ({orig:.4f})")

        ax.set_xlabel("RMSE", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(get_run_label(data["config"]).replace("\n", ", "), fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution of Shuffled RMSE vs Original", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(outdir / "shuffle_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shuffle_distributions.png")


def fig_layer_analysis(all_results: dict, outdir: Path):
    """Per-layer effect sizes for each experiment."""
    runs = sorted(all_results.keys())

    # Collect all layer names (excluding perp_combined and combined)
    sample = all_results[runs[0]]["results"]
    layer_names = [k for k in sample if k.startswith("blocks.")]
    layer_names.sort()
    layer_names.append("combined")

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(runs)), 5))

    x = np.arange(len(runs))
    width = 0.8 / len(layer_names)

    for j, layer in enumerate(layer_names):
        effects = []
        for run_name in runs:
            sc = all_results[run_name]["results"].get(layer, {})
            effects.append(sc.get("effect_size", 0) * 100)
        offset = (j - len(layer_names) / 2 + 0.5) * width
        short_name = layer.split(".")[-1] if "." in layer else layer
        ax.bar(x + offset, effects, width, label=short_name, alpha=0.8)

    labels = [get_run_label(all_results[r]["config"]).replace("\n", ", ")
              for r in runs]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel("Effect size (%)", fontsize=11)
    ax.set_title("Shuffle Control Effect by Layer", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(outdir / "shuffle_layer_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shuffle_layer_analysis.png")


def print_summary_table(all_results: dict):
    """Print a summary table to stdout."""
    print(f"\n{'Run':<45s}  {'RMSE orig':>10s}  {'RMSE shuf':>10s}  "
          f"{'Effect':>8s}  {'p-val':>6s}  {'Beyond%':>8s}  {'dim(ker)':>8s}")
    print("-" * 100)
    for run_name in sorted(all_results.keys()):
        data = all_results[run_name]
        sc = data["results"]["combined"]
        print(f"{run_name:<45s}  {sc['mse_original']:10.6f}  "
              f"{sc['mse_shuffle_mean']:10.6f}  {sc['effect_size']:7.1%}  "
              f"{sc['p_value']:6.4f}  {sc['frac_beyond']:7.1%}  "
              f"{sc['dim_kernel']:>8d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shuffle_dir", type=str, default="shuffle_results",
                        help="Directory containing *_shuffle.joblib files")
    parser.add_argument("--outdir", type=str, default="figures/shuffle",
                        help="Output directory for figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = load_shuffle_results(args.shuffle_dir)
    print(f"Loaded {len(all_results)} shuffle results from {args.shuffle_dir}")

    if not all_results:
        print("No results found. Run run_shuffle_analysis.py first.")
        return

    print_summary_table(all_results)

    fig_rmse_comparison(all_results, outdir)
    fig_effect_sizes(all_results, outdir)
    fig_shuffle_distributions(all_results, outdir)
    fig_layer_analysis(all_results, outdir)

    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
