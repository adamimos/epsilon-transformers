"""
Visualize the belief state geometry of the Quantum RRXOR process.

Generates figures showing:
  1. Default parameter 3D fractal + PCA projection
  2. Sweep over readout angle theta
  3. Sweep over memory rotation phi
  4. Sweep over leakage epsilon
  5. Non-invertibility of belief -> next-token map
  6. Variance decomposition heatmap over (phi, theta)
  7. Variance decomposition vs leakage

Usage:
    uv run python scripts/visualize_quantum_rrxor.py [--outdir figures/quantum_rrxor]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import null_space
from scipy.spatial import ConvexHull

from epsilon_transformers.process.transition_matrices import quantum_rrxor
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_belief_geometry(phi, theta, epsilon=0.0, depth=12):
    """Generate Bloch vectors and next-token probabilities."""
    T = quantum_rrxor(phi, theta, epsilon=epsilon)
    ghmm = TransitionMatrixGHMM(T)
    tree = ghmm.derive_mixed_state_tree(depth=depth)

    beliefs = np.array(tree.belief_states).squeeze()[1:]  # skip root
    bloch = beliefs[:, 1:4] / beliefs[:, 0:1]

    rev = np.array([4, 0, 0, 0])
    p0 = (beliefs @ T[0] @ rev) / (beliefs @ rev)
    return bloch, p0, T


def variance_decomposition(bloch, T):
    """Decompose belief variance into next-token-relevant and beyond."""
    alpha = np.array([T[0, 1, 0], T[0, 2, 0], T[0, 3, 0]])
    alpha_mag = np.linalg.norm(alpha)

    bloch_c = bloch - bloch.mean(axis=0)
    var_total = np.var(bloch_c)

    if alpha_mag < 1e-12 or var_total < 1e-15:
        return {"var_parallel": 0.0, "var_perp": var_total,
                "var_total": var_total,
                "frac_beyond": 1.0, "frac_next_token": 0.0,
                "alpha": alpha, "alpha_hat": np.zeros(3)}

    alpha_hat = alpha / alpha_mag
    proj_par = np.outer(bloch_c @ alpha_hat, alpha_hat)
    proj_perp = bloch_c - proj_par

    vp = np.var(proj_par)
    vo = np.var(proj_perp)
    vt = vp + vo
    return {"var_parallel": vp, "var_perp": vo, "var_total": vt,
            "frac_beyond": vo / vt, "frac_next_token": vp / vt,
            "alpha": alpha, "alpha_hat": alpha_hat}


def whitespace_metrics(bloch, n_scales=15, subsample=4000, rng=None):
    """Quantify the sparsity / whitespace of the belief geometry.

    Parameters
    ----------
    bloch : ndarray (N, 3)
        Bloch vectors of belief states.
    n_scales : int
        Number of box sizes for box-counting dimension.
    subsample : int
        Max points for expensive nearest-neighbor computation.
    rng : np.random.Generator or None
        Random generator for subsampling.

    Returns
    -------
    dict with keys:
        box_dim       : box-counting (fractal) dimension
        lacunarity    : mean lacunarity across scales (gappiness)
        mean_nn_dist  : mean nearest-neighbor distance
        hull_vol_frac : convex hull volume / Bloch ball volume
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Subsample for speed if needed
    pts = bloch
    if len(pts) > subsample:
        idx = rng.choice(len(pts), subsample, replace=False)
        pts = pts[idx]

    # --- Box-counting dimension ---
    # Normalize to [0, 1]^3 based on data range (not [-1,1] to avoid
    # counting empty space outside the fractal's extent)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = (hi - lo).max()
    if span < 1e-12:
        return {"box_dim": 0.0, "lacunarity": np.nan,
                "mean_nn_dist": 0.0, "hull_vol_frac": 0.0}
    pts_norm = (pts - lo) / span  # now in [0, 1]^3

    epsilons = np.logspace(np.log10(0.005), np.log10(0.5), n_scales)
    log_inv_eps, log_n = [], []
    lacunarities = []

    for eps in epsilons:
        # Bin points into boxes of side length eps
        bins = np.floor(pts_norm / eps).astype(int)
        # Count unique occupied boxes
        unique_boxes, counts = np.unique(bins, axis=0, return_counts=True)
        n_boxes = len(unique_boxes)
        if n_boxes > 0:
            log_inv_eps.append(np.log(1.0 / eps))
            log_n.append(np.log(n_boxes))
            # Lacunarity: Var(counts)/Mean(counts)^2 + 1
            mean_c = counts.mean()
            var_c = counts.var()
            lac = var_c / (mean_c ** 2) + 1.0 if mean_c > 0 else np.nan
            lacunarities.append(lac)

    # Fit line to log(N) vs log(1/eps) for box-counting dimension
    if len(log_inv_eps) >= 3:
        coeffs = np.polyfit(log_inv_eps, log_n, 1)
        box_dim = coeffs[0]
    else:
        box_dim = np.nan

    mean_lacunarity = np.mean(lacunarities) if lacunarities else np.nan

    # --- Mean nearest-neighbor distance ---
    nn = NearestNeighbors(n_neighbors=2).fit(pts)
    dists, _ = nn.kneighbors(pts)
    mean_nn_dist = dists[:, 1].mean()  # column 0 is self (dist=0)

    # --- Convex hull volume fraction ---
    bloch_ball_vol = (4 / 3) * np.pi  # radius = 1
    try:
        hull = ConvexHull(pts)
        hull_vol_frac = hull.volume / bloch_ball_vol
    except Exception:
        hull_vol_frac = 0.0

    return {
        "box_dim": box_dim,
        "lacunarity": mean_lacunarity,
        "mean_nn_dist": mean_nn_dist,
        "hull_vol_frac": hull_vol_frac,
    }


def plot_bloch_ball(ax, bloch, color_vals, title,
                    cmap="coolwarm", alpha=0.6, s=2.0):
    """Scatter belief states inside a wireframe Bloch sphere."""
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)),
                      np.outer(np.sin(u), np.sin(v)),
                      np.outer(np.ones_like(u), np.cos(v)),
                      color="gray", alpha=0.05, linewidth=0.3)
    sc = ax.scatter(bloch[:, 0], bloch[:, 1], bloch[:, 2],
                    c=color_vals, cmap=cmap, s=s, alpha=alpha,
                    edgecolors="none")
    ax.set(xlabel="$b_x$", ylabel="$b_y$", zlabel="$b_z$",
           xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), title=title)
    ax.tick_params(labelsize=6)
    for lbl in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        lbl.set_fontsize(8)
    ax.title.set_fontsize(10)
    return sc


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_default(outdir):
    bloch, p0, T = get_belief_geometry(1.0, np.pi / 3, depth=14)
    vd = variance_decomposition(bloch, T)

    fig = plt.figure(figsize=(14, 5))
    for i, (elev, azim) in enumerate([(25, -60), (25, 30)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        plot_bloch_ball(ax, bloch, p0,
                        f"$\\varphi=1.0$, $\\theta=\\pi/3$ (view {i+1})",
                        alpha=0.6, s=1.5)
        ax.view_init(elev=elev, azim=azim)

    ax2d = fig.add_subplot(1, 3, 3)
    pca = PCA(n_components=2).fit(bloch)
    proj = pca.transform(bloch)
    sc = ax2d.scatter(proj[:, 0], proj[:, 1], c=p0, cmap="coolwarm",
                      s=0.8, alpha=0.4, edgecolors="none")
    ax2d.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
             ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
             title="PCA projection, colored by $p(0)$", aspect="equal")
    ax2d.tick_params(labelsize=6)
    fig.colorbar(sc, ax=ax2d, label="$p(\\mathrm{token}=0)$", shrink=0.7)
    fig.suptitle("Quantum RRXOR — default parameters", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(outdir / "01_default_params.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pca3 = PCA().fit(bloch)
    print(f"  beliefs={len(bloch)}, PCA=[{pca3.explained_variance_ratio_[0]:.1%},"
          f" {pca3.explained_variance_ratio_[1]:.1%},"
          f" {pca3.explained_variance_ratio_[2]:.1%}],"
          f" p(0)=[{p0.min():.4f},{p0.max():.4f}],"
          f" next-tok={vd['frac_next_token']:.1%},"
          f" beyond={vd['frac_beyond']:.1%}")


def fig_theta_sweep(outdir):
    theta_vals = [np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2,
                  5*np.pi/8, 3*np.pi/4, 7*np.pi/8]
    theta_lbls = ["\\pi/8", "\\pi/4", "3\\pi/8", "\\pi/2",
                  "5\\pi/8", "3\\pi/4", "7\\pi/8"]

    fig = plt.figure(figsize=(24, 4))
    for i, (th, lbl) in enumerate(zip(theta_vals, theta_lbls)):
        b, p, _ = get_belief_geometry(1.0, th, depth=13)
        ax = fig.add_subplot(1, 7, i + 1, projection="3d")
        plot_bloch_ball(ax, b, p, f"$\\theta = {lbl}$", s=1.0, alpha=0.5)
        ax.view_init(elev=25, azim=-60)
    fig.suptitle("Varying readout angle $\\theta$  ($\\varphi = 1.0$, $\\varepsilon = 0$)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(outdir / "02_theta_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_phi_sweep(outdir):
    phi_vals = [0.1, 0.5, 1.0, 1.5, 2.0, np.pi, 4.0]
    phi_lbls = ["0.1", "0.5", "1.0", "1.5", "2.0", "\\pi", "4.0"]

    fig = plt.figure(figsize=(24, 4))
    for i, (ph, lbl) in enumerate(zip(phi_vals, phi_lbls)):
        b, p, _ = get_belief_geometry(ph, np.pi / 3, depth=13)
        ax = fig.add_subplot(1, 7, i + 1, projection="3d")
        plot_bloch_ball(ax, b, p, f"$\\varphi = {lbl}$", s=1.0, alpha=0.5)
        ax.view_init(elev=25, azim=-60)
    fig.suptitle("Varying memory rotation $\\varphi$  ($\\theta = \\pi/3$, $\\varepsilon = 0$)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(outdir / "03_phi_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_eps_sweep(outdir):
    eps_vals = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4]

    fig = plt.figure(figsize=(24, 4))
    for i, eps in enumerate(eps_vals):
        b, p, _ = get_belief_geometry(1.0, np.pi / 3, epsilon=eps, depth=13)
        ax = fig.add_subplot(1, 7, i + 1, projection="3d")
        plot_bloch_ball(ax, b, p, f"$\\varepsilon = {eps}$", s=1.0, alpha=0.5)
        ax.view_init(elev=25, azim=-60)
    fig.suptitle("Varying leakage $\\varepsilon$  ($\\varphi = 1.0$, $\\theta = \\pi/3$)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(outdir / "04_eps_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_noninvertibility(outdir):
    bloch, p0, T = get_belief_geometry(1.0, np.pi / 3, depth=14)

    reg = LinearRegression().fit(bloch, p0)
    alpha = reg.coef_
    alpha_hat = alpha / np.linalg.norm(alpha)
    kb = null_space(alpha.reshape(1, -1))
    u1, u2 = kb[:, 0], kb[:, 1]

    print(f"  p(0|b) = {reg.intercept_:.4f} + {alpha[0]:.4f}*b_x"
          f" + {alpha[1]:.4f}*b_y + {alpha[2]:.4f}*b_z")
    print(f"  R² = {reg.score(bloch, p0):.10f}")
    print(f"  alpha_hat = ({alpha_hat[0]:.4f}, {alpha_hat[1]:.4f}, {alpha_hat[2]:.4f})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, proj, title in [
        (axes[0], bloch @ alpha_hat,
         "Next-token direction vs $p(0)$\n(perfect correlation)"),
        (axes[1], bloch @ u1,
         "Kernel direction 1 vs $p(0)$\n(no correlation)"),
        (axes[2], bloch @ u2,
         "Kernel direction 2 vs $p(0)$\n(no correlation)"),
    ]:
        ax.scatter(proj, p0, c=p0, cmap="coolwarm", s=0.5, alpha=0.4)
        ax.set_ylabel("$p(0)$", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)
    axes[0].set_xlabel("Projection onto $\\hat{\\alpha}$ (next-token direction)", fontsize=8)
    axes[1].set_xlabel("Projection onto kernel direction $u_1$", fontsize=8)
    axes[2].set_xlabel("Projection onto kernel direction $u_2$", fontsize=8)
    fig.suptitle("Non-invertibility of the belief $\\to$ next-token map",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(outdir / "05_noninvertibility.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_variance_heatmap(outdir):
    phi_grid = np.linspace(0.1, 4.0, 20)
    theta_grid = np.linspace(np.pi / 8, 7 * np.pi / 8, 20)

    frac_grid = np.zeros((len(phi_grid), len(theta_grid)))
    p0r_grid = np.zeros_like(frac_grid)

    for i, ph in enumerate(phi_grid):
        for j, th in enumerate(theta_grid):
            b, p, t = get_belief_geometry(ph, th, depth=11)
            vd = variance_decomposition(b, t)
            frac_grid[i, j] = vd["frac_beyond"]
            p0r_grid[i, j] = p.max() - p.min()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(frac_grid.T, origin="lower", aspect="auto",
                          extent=[phi_grid[0], phi_grid[-1],
                                  theta_grid[0], theta_grid[-1]],
                          cmap="viridis", vmin=0, vmax=1)
    axes[0].set(xlabel="$\\varphi$ (memory rotation)",
                ylabel="$\\theta$ (readout angle)",
                title="Fraction of belief variance\nbeyond next-token prediction")
    fig.colorbar(im0, ax=axes[0],
                 label="$\\mathrm{Var}(b^\\perp) / \\mathrm{Var}(b)$", shrink=0.8)

    im1 = axes[1].imshow(p0r_grid.T, origin="lower", aspect="auto",
                          extent=[phi_grid[0], phi_grid[-1],
                                  theta_grid[0], theta_grid[-1]],
                          cmap="magma")
    axes[1].set(xlabel="$\\varphi$ (memory rotation)",
                ylabel="$\\theta$ (readout angle)",
                title="Dynamic range of $p(0)$\n(max $-$ min)")
    fig.colorbar(im1, ax=axes[1],
                 label="$p(0)_{\\max} - p(0)_{\\min}$", shrink=0.8)

    for ax in axes:
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(outdir / "06_variance_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print table of key values
    print(f"  {'phi':>6s}  {'theta':>8s}  {'frac_beyond':>12s}  {'p0_range':>10s}")
    print("  " + "-" * 42)
    for ph, th in [(1.0, np.pi/3), (1.0, np.pi/2), (1.0, 3*np.pi/4),
                   (0.5, np.pi/3), (2.0, np.pi/3), (np.pi, np.pi/2),
                   (3.0, np.pi/4), (0.1, np.pi/2)]:
        b, p, t = get_belief_geometry(ph, th, depth=12)
        vd = variance_decomposition(b, t)
        print(f"  {ph:6.2f}  {th:8.4f}  {vd['frac_beyond']:11.1%}  "
              f"{p.max()-p.min():10.4f}")


def fig_whitespace_heatmap(outdir):
    phi_grid = np.linspace(0.1, 4.0, 15)
    theta_grid = np.linspace(np.pi / 8, 7 * np.pi / 8, 15)

    box_dim_grid = np.zeros((len(phi_grid), len(theta_grid)))
    lacunarity_grid = np.zeros_like(box_dim_grid)
    nn_dist_grid = np.zeros_like(box_dim_grid)
    hull_grid = np.zeros_like(box_dim_grid)

    rng = np.random.default_rng(42)
    for i, ph in enumerate(phi_grid):
        for j, th in enumerate(theta_grid):
            b, _, _ = get_belief_geometry(ph, th, depth=11)
            wm = whitespace_metrics(b, rng=rng)
            box_dim_grid[i, j] = wm["box_dim"]
            lacunarity_grid[i, j] = wm["lacunarity"]
            nn_dist_grid[i, j] = wm["mean_nn_dist"]
            hull_grid[i, j] = wm["hull_vol_frac"]

    ext = [phi_grid[0], phi_grid[-1], theta_grid[0], theta_grid[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    data_and_labels = [
        (box_dim_grid, "Box-counting dimension", "viridis", None),
        (lacunarity_grid, "Lacunarity (gappiness)", "inferno", None),
        (nn_dist_grid, "Mean nearest-neighbor distance", "cividis", None),
        (hull_grid, "Convex hull volume / Bloch ball", "magma", None),
    ]

    for ax, (grid, title, cmap, vlim) in zip(axes.flat, data_and_labels):
        kw = {"origin": "lower", "aspect": "auto", "extent": ext, "cmap": cmap}
        if vlim:
            kw["vmin"], kw["vmax"] = vlim
        im = ax.imshow(grid.T, **kw)
        ax.set(xlabel="$\\varphi$ (memory rotation)",
               ylabel="$\\theta$ (readout angle)", title=title)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.tick_params(labelsize=8)

    fig.suptitle("Whitespace metrics across parameter space", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(outdir / "08_whitespace_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print table for the 8 proposed experiments
    print(f"  {'phi':>5s}  {'theta':>8s}  {'eps':>5s}  {'box_dim':>8s}  "
          f"{'lacunar':>8s}  {'nn_dist':>8s}  {'hull_frac':>10s}  {'beyond%':>8s}")
    print("  " + "-" * 72)
    expts = [(1.0, np.pi/3, 0.05), (2.0, np.pi/3, 0.05),
             (0.5, np.pi/3, 0.05), (1.0, 2*np.pi/3, 0.05),
             (3.0, np.pi/4, 0.05), (1.0, np.pi/3, 0.0),
             (1.0, np.pi/3, 0.10), (1.0, 7*np.pi/8, 0.05)]
    for ph, th, eps in expts:
        b, p, t = get_belief_geometry(ph, th, epsilon=eps, depth=12)
        vd = variance_decomposition(b, t)
        wm = whitespace_metrics(b, rng=rng)
        print(f"  {ph:5.2f}  {th:8.4f}  {eps:5.2f}  {wm['box_dim']:8.2f}  "
              f"{wm['lacunarity']:8.2f}  {wm['mean_nn_dist']:8.4f}  "
              f"{wm['hull_vol_frac']:10.4f}  {vd['frac_beyond']:7.1%}")


def fig_variance_vs_leakage(outdir):
    eps_sweep = np.linspace(0, 0.5, 30)
    phis = [0.5, 1.0, 2.0, np.pi]
    labels = ["0.5", "1.0", "2.0", "\\pi"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ph, lbl in zip(phis, labels):
        fracs, tots = [], []
        for eps in eps_sweep:
            b, _, t = get_belief_geometry(ph, np.pi / 3, epsilon=eps, depth=11)
            vd = variance_decomposition(b, t)
            fracs.append(vd["frac_beyond"])
            tots.append(vd["var_total"])
        axes[0].plot(eps_sweep, fracs, label=f"$\\varphi = {lbl}$", lw=2)
        axes[1].plot(eps_sweep, tots, label=f"$\\varphi = {lbl}$", lw=2)

    axes[0].set(xlabel="$\\varepsilon$ (leakage)",
                ylabel="Fraction beyond next-token", ylim=(0, 1),
                title="Beyond-next-token variance fraction\nvs leakage")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].set(xlabel="$\\varepsilon$ (leakage)",
                ylabel="Total variance",
                title="Total belief variance vs leakage\n(log scale)")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "07_variance_vs_leakage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=str, default="figures/quantum_rrxor",
                        help="Directory for output PNGs")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runners = [
        ("01 Default parameters", fig_default),
        ("02 Theta sweep", fig_theta_sweep),
        ("03 Phi sweep", fig_phi_sweep),
        ("04 Epsilon sweep", fig_eps_sweep),
        ("05 Non-invertibility", fig_noninvertibility),
        ("06 Variance heatmap", fig_variance_heatmap),
        ("07 Variance vs leakage", fig_variance_vs_leakage),
        ("08 Whitespace metrics", fig_whitespace_heatmap),
    ]

    for name, fn in runners:
        print(f"[{name}]")
        fn(outdir)

    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
