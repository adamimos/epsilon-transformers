"""
Geometric Shuffle Control for Belief State Regression.

Tests whether neural networks encode belief structure *beyond* local next-token
prediction, by decomposing beliefs into a next-token-relevant component and an
orthogonal component, shuffling the orthogonal part, and comparing regression quality.

Works for both classical HMMs and GHMMs (quantum / post-quantum processes).

Reference: context/shufflecontrol.md
"""
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Emission matrix and projections
# ---------------------------------------------------------------------------

def compute_emission_matrix(T: np.ndarray, rev: np.ndarray) -> np.ndarray:
    """Compute the emission matrix E from GHMM transition matrices.

    E[:, x] = T[x] @ rev, so that p(x | eta) = eta @ E[:, x] for normalized
    belief states (eta @ rev = 1).

    For classical HMMs (rev = ones), this reduces to E[s, x] = sum_j T[x, s, j].

    Parameters
    ----------
    T : ndarray, shape (vocab_len, latent_dim, latent_dim)
        GHMM transition matrices.
    rev : ndarray, shape (latent_dim,) or (latent_dim, 1)
        Right eigenvector of T.sum(axis=0) with eigenvalue 1.

    Returns
    -------
    E : ndarray, shape (latent_dim, vocab_len)
    """
    rev = rev.squeeze()
    vocab_len, latent_dim = T.shape[0], T.shape[1]
    E = np.zeros((latent_dim, vocab_len))
    for x in range(vocab_len):
        E[:, x] = T[x] @ rev
    return E


def compute_projections(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute orthogonal projections onto col(E) and ker(E^T).

    P_E projects onto col(E):      the subspace relevant for next-token prediction.
    P_E_perp projects onto ker(E^T): the orthogonal subspace (beyond next-token).

    Parameters
    ----------
    E : ndarray, shape (latent_dim, vocab_len)

    Returns
    -------
    P_E : ndarray, shape (latent_dim, latent_dim)
    P_E_perp : ndarray, shape (latent_dim, latent_dim)
    """
    latent_dim = E.shape[0]
    ETE = E.T @ E
    ETE_inv = np.linalg.pinv(ETE)
    P_E = E @ ETE_inv @ E.T
    P_E_perp = np.eye(latent_dim) - P_E
    return P_E, P_E_perp


# ---------------------------------------------------------------------------
# Belief decomposition and shuffling
# ---------------------------------------------------------------------------

def decompose_beliefs(
    beliefs: np.ndarray,
    P_E: np.ndarray,
    P_E_perp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose beliefs into next-token-relevant and orthogonal components.

    Parameters
    ----------
    beliefs : ndarray, shape (N, latent_dim)
    P_E, P_E_perp : ndarray, shape (latent_dim, latent_dim)

    Returns
    -------
    b_mean : ndarray, shape (latent_dim,)
    delta_parallel : ndarray, shape (N, latent_dim)  — in col(E)
    delta_perp : ndarray, shape (N, latent_dim)      — in ker(E^T)
    """
    b_mean = beliefs.mean(axis=0)
    delta = beliefs - b_mean
    # P_E and P_E_perp are symmetric, so left- and right-multiply are equivalent
    delta_parallel = delta @ P_E
    delta_perp = delta @ P_E_perp
    return b_mean, delta_parallel, delta_perp


def geometric_shuffle(
    beliefs: np.ndarray,
    P_E: np.ndarray,
    P_E_perp: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct shuffled belief targets.

    Preserves the next-token component (delta_parallel) and randomizes the
    beyond-next-token component (delta_perp) via a random permutation.

    Parameters
    ----------
    beliefs : ndarray, shape (N, latent_dim)
    P_E, P_E_perp : ndarray, shape (latent_dim, latent_dim)
    rng : numpy random Generator

    Returns
    -------
    shuffled : ndarray, shape (N, latent_dim)
    """
    b_mean, delta_parallel, delta_perp = decompose_beliefs(beliefs, P_E, P_E_perp)
    perm = rng.permutation(len(beliefs))
    return b_mean + delta_parallel + delta_perp[perm]


# ---------------------------------------------------------------------------
# Regression (weighted least squares, matching the paper's procedure)
# ---------------------------------------------------------------------------

def _weighted_r2(X, Y, weights, rcond=1e-10):
    """Weighted least-squares regression, returns R².

    Parameters
    ----------
    X : ndarray, shape (N, d_features)
    Y : ndarray, shape (N, d_targets)
    weights : ndarray, shape (N,)
    rcond : float — regularization for lstsq

    Returns
    -------
    r2 : float — weighted R² (coefficient of determination)
    """
    N = X.shape[0]
    w = weights / weights.sum()
    sqrt_w = np.sqrt(w)[:, None]

    # Add bias
    X_bias = np.hstack([np.ones((N, 1)), X])

    # Weighted regression
    X_w = X_bias * sqrt_w
    Y_w = Y * sqrt_w

    # Solve
    beta, _, _, _ = np.linalg.lstsq(X_w, Y_w, rcond=rcond)

    # Predict (unweighted)
    Y_pred = X_bias @ beta

    # Weighted R²
    Y_mean = (Y * w[:, None]).sum(axis=0)  # weighted mean
    ss_res = (w[:, None] * (Y - Y_pred) ** 2).sum()
    ss_tot = (w[:, None] * (Y - Y_mean) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(r2)


# ---------------------------------------------------------------------------
# Full shuffle control experiment
# ---------------------------------------------------------------------------

def run_shuffle_control(
    activations: np.ndarray,
    beliefs: np.ndarray,
    T: np.ndarray,
    rev: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_shuffles: int = 100,
    seed: int = 42,
    show_progress: bool = True,
) -> Dict:
    """Run the geometric shuffle control experiment.

    Parameters
    ----------
    activations : ndarray, shape (N, d_features)
        Neural network activations (e.g., residual stream at last token position).
    beliefs : ndarray, shape (N, latent_dim)
        Ground-truth belief states.
    T : ndarray, shape (vocab_len, latent_dim, latent_dim)
        GHMM transition matrices.
    rev : ndarray, shape (latent_dim,) or (latent_dim, 1)
        Right eigenvector of T.sum(axis=0).
    weights : ndarray, shape (N,), optional
        Probability weights for each sequence. Uniform if None.
    n_shuffles : int
        Number of random shuffles for the null distribution.
    seed : int
        Random seed for reproducibility.
    show_progress : bool
        Show progress bar.

    Returns
    -------
    results : dict with keys:
        mse_original    : float — RMSE with true beliefs
        mse_shuffled    : list[float] — RMSE for each shuffle
        mse_shuffle_mean: float — mean shuffled RMSE
        mse_shuffle_std : float — std of shuffled RMSE
        effect_size     : float — (mean_shuffled - original) / original
        p_value         : float — fraction of shuffles with RMSE <= original
        var_parallel    : float — variance in next-token subspace
        var_perp        : float — variance in beyond-next-token subspace
        frac_beyond     : float — var_perp / (var_parallel + var_perp)
        dim_kernel      : int — dimension of ker(E^T)
        E               : ndarray — emission matrix
        P_E             : ndarray — projection onto col(E)
        P_E_perp        : ndarray — projection onto ker(E^T)
    """
    rng = np.random.default_rng(seed)

    if weights is None:
        weights = np.ones(len(beliefs)) / len(beliefs)

    # Compute emission matrix and projections
    E = compute_emission_matrix(T, rev)
    P_E, P_E_perp = compute_projections(E)

    rank_E = np.linalg.matrix_rank(E)
    dim_kernel = beliefs.shape[1] - rank_E

    # Decompose beliefs for variance diagnostics
    b_mean, delta_par, delta_perp = decompose_beliefs(beliefs, P_E, P_E_perp)
    var_par = np.var(delta_par)
    var_perp = np.var(delta_perp)
    var_total = var_par + var_perp

    # Original regression
    mse_original = _weighted_r2(activations, beliefs, weights)

    # Shuffled regressions
    mse_shuffled = []
    iterator = range(n_shuffles)
    if show_progress:
        iterator = tqdm(iterator, desc="Shuffle control", leave=False)

    for _ in iterator:
        shuffled = geometric_shuffle(beliefs, P_E, P_E_perp, rng)
        mse = _weighted_r2(activations, shuffled, weights)
        mse_shuffled.append(mse)

    mse_shuffled = np.array(mse_shuffled)

    return {
        "mse_original": mse_original,
        "mse_shuffled": mse_shuffled.tolist(),
        "mse_shuffle_mean": float(mse_shuffled.mean()),
        "mse_shuffle_std": float(mse_shuffled.std()),
        "effect_size": float((mse_shuffled.mean() - mse_original) / mse_original)
            if mse_original > 0 else 0.0,
        "p_value": float(np.mean(mse_shuffled <= mse_original)),
        "var_parallel": float(var_par),
        "var_perp": float(var_perp),
        "frac_beyond": float(var_perp / var_total) if var_total > 0 else 0.0,
        "dim_kernel": int(dim_kernel),
        "E": E,
        "P_E": P_E,
        "P_E_perp": P_E_perp,
    }


# ---------------------------------------------------------------------------
# Direct perpendicular regression (supplementary test)
# ---------------------------------------------------------------------------

def run_perpendicular_regression(
    activations: np.ndarray,
    beliefs: np.ndarray,
    T: np.ndarray,
    rev: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_shuffles: int = 100,
    seed: int = 42,
) -> Dict:
    """Directly regress activations onto the beyond-next-token component.

    If the network encodes beyond-next-token structure, regression onto
    delta_perp should achieve much lower RMSE than a shuffled baseline.

    Parameters
    ----------
    activations, beliefs, T, rev, weights, n_shuffles, seed : same as run_shuffle_control

    Returns
    -------
    results : dict with keys:
        mse_perp_original : float — RMSE regressing to delta_perp
        mse_perp_shuffled : list[float] — RMSE with shuffled delta_perp
        effect_size : float
        p_value : float
    """
    rng = np.random.default_rng(seed)

    if weights is None:
        weights = np.ones(len(beliefs)) / len(beliefs)

    E = compute_emission_matrix(T, rev)
    P_E, P_E_perp = compute_projections(E)
    _, _, delta_perp = decompose_beliefs(beliefs, P_E, P_E_perp)

    # Regress activations onto perpendicular component
    mse_orig = _weighted_r2(activations, delta_perp, weights)

    # Shuffled baseline
    mse_shuffled = []
    for _ in range(n_shuffles):
        perm = rng.permutation(len(delta_perp))
        mse = _weighted_r2(activations, delta_perp[perm], weights)
        mse_shuffled.append(mse)

    mse_shuffled = np.array(mse_shuffled)

    return {
        "mse_perp_original": float(mse_orig),
        "mse_perp_shuffled": mse_shuffled.tolist(),
        "effect_size": float((mse_shuffled.mean() - mse_orig) / mse_orig)
            if mse_orig > 0 else 0.0,
        "p_value": float(np.mean(mse_shuffled <= mse_orig)),
    }
