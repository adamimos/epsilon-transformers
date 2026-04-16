"""
Figure: Geometric Shuffle Control Results.

Primary figure: 5 curves + random baseline across layers showing that
transformers encode belief structure beyond next-token prediction.

Also generates training dynamics (same curves across checkpoints) and
shuffle distribution histograms.

Usage:
    uv run python FigShuffle.py --shuffle_dir shuffle_results
"""
import argparse
import os
import glob
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


CURVE_STYLES = {
    "acts_to_beliefs":     {"color": "#1565C0", "ls": "-",  "lw": 2.8, "marker": "o",
                            "label": "Activations $\\to$ beliefs", "zorder": 10, "ms": 7},
    "acts_to_shuffled":    {"color": "#C62828", "ls": "-",  "lw": 2.8, "marker": "s",
                            "label": "Activations $\\to$ shuffled", "zorder": 9, "ms": 7},
    "ntp_to_beliefs":      {"color": "#7B1FA2", "ls": "--", "lw": 1.8, "marker": "^",
                            "label": "Next-token $\\to$ beliefs", "zorder": 5, "ms": 6},
    "beliefs_to_shuffled": {"color": "#E65100", "ls": "--", "lw": 1.8, "marker": "d",
                            "label": "Beliefs $\\to$ shuffled", "zorder": 5, "ms": 6},
    "ntp_to_shuffled":     {"color": "#2E7D32", "ls": "--", "lw": 1.8, "marker": "v",
                            "label": "Next-token $\\to$ shuffled", "zorder": 5, "ms": 6},
    "random_to_beliefs":   {"color": "#757575", "ls": ":",  "lw": 1.8, "marker": "x",
                            "label": "Random $\\to$ beliefs", "zorder": 4, "ms": 7},
}


def load_all_results(shuffle_dir: str) -> dict:
    results = {}
    for fpath in sorted(glob.glob(os.path.join(shuffle_dir, "*_shuffle.joblib"))):
        data = joblib.load(fpath)
        run_name = os.path.basename(fpath).replace("_shuffle.joblib", "")
        results[run_name] = data
    return results


def get_run_label(config: dict) -> str:
    pc = config["process_config"]
    return (f"$\\varphi$={pc.get('phi',0):.1f}, "
            f"$\\theta$={pc.get('theta',0):.2f}, "
            f"$\\varepsilon$={pc.get('epsilon',0):.2f}")


def _sort_key(k):
    if "resid_pre" in k:
        return (0, int(k.split(".")[1]))
    elif "resid_post" in k and k.startswith("blocks."):
        return (1, int(k.split(".")[1]))
    elif "ln_final" in k:
        return (2, 0)
    elif k == "logits":
        return (3, 0)
    return (4, 0)


def nice_layer_labels(layer_keys):
    """Convert hook names to readable labels."""
    labels = []
    for k in layer_keys:
        if "resid_pre" in k:
            idx = k.split(".")[1]
            labels.append(f"L{idx} pre")
        elif "resid_post" in k and k.startswith("blocks."):
            idx = k.split(".")[1]
            labels.append(f"L{idx} post")
        elif "ln_final" in k:
            labels.append("LN final")
        elif k == "logits":
            labels.append("Logits")
        else:
            labels.append(k)
    return labels


# ---------------------------------------------------------------------------
# Main figure: 5 curves + random baseline across layers
# ---------------------------------------------------------------------------

def fig_layer_curves(data: dict, outdir: Path, run_name: str):
    """The primary figure: R² vs layer for all 6 regression curves."""
    ckpt = data["checkpoint_results"][-1]
    curves = ckpt["curves"]

    all_layers = list(curves.keys())
    layer_order = [k for k in all_layers if k != "combined"]
    layer_order.sort(key=_sort_key)
    x = np.arange(len(layer_order))
    labels = nice_layer_labels(layer_order)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Shade the gap between acts→beliefs and acts→shuffled
    vals_bel = [curves[l]["acts_to_beliefs"] for l in layer_order]
    vals_shuf = [curves[l]["acts_to_shuffled"] for l in layer_order]
    ax.fill_between(x, vals_shuf, vals_bel, alpha=0.12, color="#1565C0",
                    label="_nolegend_")

    for curve_name, style in CURVE_STYLES.items():
        vals = [curves[layer][curve_name] for layer in layer_order]
        ax.plot(x, vals, color=style["color"], ls=style["ls"], lw=style["lw"],
                marker=style["marker"], markersize=style["ms"],
                label=style["label"], zorder=style["zorder"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="medium")
    ax.set_xlabel("Layer", fontsize=12, fontweight="medium")
    ax.set_ylabel("$R^2$ (variance explained)", fontsize=12, fontweight="medium")
    ax.tick_params(axis="y", labelsize=10)

    # Title with process params
    pc = data["config"]["process_config"]
    title = (f"Quantum RRXOR: $\\varphi$={pc.get('phi',0):.1f}, "
             f"$\\theta$={pc.get('theta',0):.2f}, "
             f"$\\varepsilon$={pc.get('epsilon',0):.2f}")
    ax.set_title(title, fontsize=12, pad=12)

    ax.legend(fontsize=9, loc="upper left", framealpha=0.92,
              edgecolor="0.8", fancybox=False)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(outdir / f"layer_curves_{run_name}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved layer_curves_{run_name}.png")


def fig_layer_curves_grid(all_results: dict, outdir: Path):
    """Grid of layer-curve plots for all experiments."""
    n = len(all_results)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, (run_name, data) in enumerate(sorted(all_results.items())):
        ax = axes[idx // ncols][idx % ncols]
        ckpt = data["checkpoint_results"][-1]
        curves = ckpt["curves"]

        all_layers = list(curves.keys())
        layer_order = [k for k in all_layers if k != "combined"]
        layer_order.sort(key=_sort_key)
        x = np.arange(len(layer_order))
        labels = nice_layer_labels(layer_order)

        for curve_name, style in CURVE_STYLES.items():
            vals = [curves[layer][curve_name] for layer in layer_order]
            ax.plot(x, vals, color=style["color"], ls=style["ls"],
                    lw=style["lw"], marker=style["marker"], markersize=4,
                    label=style["label"] if idx == 0 else None)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30)
        ax.set_title(get_run_label(data["config"]), fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylabel("$R^2$", fontsize=8)

    # Hide empty subplots
    for idx in range(len(all_results), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend
    handles, labels_leg = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Shuffle Control: RMSE by Layer", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(outdir / "layer_curves_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved layer_curves_grid.png")


# ---------------------------------------------------------------------------
# Training dynamics: curves across checkpoints
# ---------------------------------------------------------------------------

def fig_training_dynamics(data: dict, outdir: Path, run_name: str):
    """5 curves + random baseline across training checkpoints (last resid layer)."""
    ckpts = data["checkpoint_results"]
    if len(ckpts) < 3:
        print(f"  Skipping training dynamics (only {len(ckpts)} checkpoints)")
        return

    tokens = [c["tokens_seen"] for c in ckpts]

    # Find the last resid_post layer
    sample_curves = ckpts[0]["curves"]
    last_layer = [k for k in sample_curves if "resid_post" in k and k.startswith("blocks.")]
    last_layer = last_layer[-1] if last_layer else list(sample_curves.keys())[-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    for curve_name, style in CURVE_STYLES.items():
        vals = [c["curves"][last_layer][curve_name] for c in ckpts]
        ax.plot(tokens, vals, color=style["color"], ls=style["ls"],
                lw=style["lw"], marker=style["marker"], markersize=4,
                label=style["label"])

    ax.set_xlabel("Tokens seen", fontsize=11)
    ax.set_ylabel("$R^2$ (final layer)", fontsize=11)
    ax.set_title(f"Training Dynamics — {get_run_label(data['config'])}",
                 fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(outdir / f"training_dynamics_{run_name}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training_dynamics_{run_name}.png")


# ---------------------------------------------------------------------------
# Shuffle distribution histogram
# ---------------------------------------------------------------------------

def fig_shuffle_distributions(all_results: dict, outdir: Path):
    """Histogram of shuffled RMSE vs original for each experiment."""
    n = len(all_results)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, (run_name, data) in enumerate(sorted(all_results.items())):
        ax = axes[idx // ncols][idx % ncols]
        sd = data["checkpoint_results"][-1]["shuffle_dist"]

        shuffled = np.array(sd["r2_shuffled"])
        orig = sd["r2_original"]

        ax.hist(shuffled, bins=30, color="#F44336", alpha=0.7,
                edgecolor="black", linewidth=0.5, label="Shuffled")
        ax.axvline(orig, color="#2196F3", linewidth=2, linestyle="--",
                   label=f"Original ({orig:.3f})")
        ax.set_xlabel("$R^2$", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(get_run_label(data["config"]), fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for idx in range(len(all_results), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Distribution of Shuffled RMSE vs Original", fontsize=13,
                 y=1.01)
    plt.tight_layout()
    fig.savefig(outdir / "shuffle_distributions.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shuffle_distributions.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: dict):
    print(f"\n{'Run':<45s}  {'acts→bel':>9s}  {'acts→shuf':>10s}  "
          f"{'ntp→bel':>8s}  {'rand→bel':>9s}  {'p-val':>6s}")
    print("-" * 95)
    for name in sorted(all_results.keys()):
        data = all_results[name]
        # Use last resid_post layer
        curves = data["checkpoint_results"][-1]["curves"]
        layer_keys = list(curves.keys())
        last_layer = [k for k in layer_keys if "resid_post" in k][-1]
        c = curves[last_layer]
        sd = data["checkpoint_results"][-1]["shuffle_dist"]
        print(f"{name:<45s}  {c['acts_to_beliefs']:9.4f}  "
              f"{c['acts_to_shuffled']:10.4f}  {c['ntp_to_beliefs']:8.4f}  "
              f"{c['random_to_beliefs']:9.4f}  {sd['p_value']:6.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--shuffle_dir", type=str, default="shuffle_results")
    parser.add_argument("--outdir", type=str, default="figures/shuffle")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = load_all_results(args.shuffle_dir)
    print(f"Loaded {len(all_results)} results from {args.shuffle_dir}")

    if not all_results:
        print("No results found.")
        return

    print_summary(all_results)

    # Per-run figures
    for run_name, data in sorted(all_results.items()):
        fig_layer_curves(data, outdir, run_name)
        fig_training_dynamics(data, outdir, run_name)

    # Cross-run figures
    fig_layer_curves_grid(all_results, outdir)
    fig_shuffle_distributions(all_results, outdir)

    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
