"""
Geometric Shuffle Control for Belief State Regression.

Tests whether neural networks encode belief structure *beyond* local next-token
prediction, by decomposing beliefs into a next-token-relevant component and an
orthogonal component, shuffling the orthogonal part, and comparing regression quality.

Uses the same regression procedure as the paper (SVD-based pseudoinverse with
cross-validated rcond selection) for consistency.

Works for both classical HMMs and GHMMs (quantum / post-quantum processes).

Reference: context/shufflecontrol.md
"""
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from tqdm.auto import tqdm

from scripts.activation_analysis.regression import run_activation_to_beliefs_regression_kf
from scripts.activation_analysis.config import RCOND_SWEEP_LIST


# ---------------------------------------------------------------------------
# Emission matrix and projections
# ---------------------------------------------------------------------------

def compute_emission_matrix(T: np.ndarray, rev: np.ndarray) -> np.ndarray:
    """Compute the emission matrix E from GHMM transition matrices.

    E[:, x] = T[x] @ rev, so that p(x | eta) = eta @ E[:, x] for normalized
    belief states (eta @ rev = 1).

    For classical HMMs (rev = ones), this reduces to E[s, x] = sum_j T[x, s, j].
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

def decompose_beliefs(beliefs, P_E, P_E_perp):
    """Decompose beliefs into next-token-relevant and orthogonal components."""
    b_mean = beliefs.mean(axis=0)
    delta = beliefs - b_mean
    delta_parallel = delta @ P_E
    delta_perp = delta @ P_E_perp
    return b_mean, delta_parallel, delta_perp


def geometric_shuffle(beliefs, P_E, P_E_perp, rng):
    """Shuffle beyond-next-token component while preserving next-token component."""
    b_mean, delta_parallel, delta_perp = decompose_beliefs(beliefs, P_E, P_E_perp)
    perm = rng.permutation(len(beliefs))
    return b_mean + delta_parallel + delta_perp[perm]


# ---------------------------------------------------------------------------
# Regression using the paper's procedure
# ---------------------------------------------------------------------------

def _paper_regression_r2(
    activations_np: np.ndarray,
    beliefs_np: np.ndarray,
    weights_np: np.ndarray,
    n_splits: int = 10,
    device: str = "cpu",
) -> float:
    """Run the paper's regression procedure and return R².

    Uses run_activation_to_beliefs_regression_kf with SVD-based pseudoinverse
    and cross-validated rcond selection, matching the paper exactly.

    Returns R² (computed from the final model's predictions).
    """
    # Convert to torch tensors
    acts_t = torch.tensor(activations_np, dtype=torch.float32, device=device)
    bel_t = torch.tensor(beliefs_np, dtype=torch.float32, device=device)
    w_t = torch.tensor(weights_np, dtype=torch.float32, device=device)

    # Create k-fold splits
    N = len(activations_np)
    indices = np.arange(N)
    fold_size = N // n_splits
    kf = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_splits - 1 else N
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        kf.append((train_idx, test_idx))

    # Run the paper's regression
    result = run_activation_to_beliefs_regression_kf(
        regression_analyzer=None,
        activations=acts_t,
        beliefs=bel_t,
        probs=w_t,
        kf=kf,
        rcond_values=RCOND_SWEEP_LIST,
    )

    # Compute R² from predictions
    predictions = result.get("predictions")
    if predictions is None:
        return 0.0

    predictions = np.asarray(predictions)
    w = weights_np / weights_np.sum()
    Y_mean = (beliefs_np * w[:, None]).sum(axis=0)
    ss_res = (w[:, None] * (beliefs_np - predictions) ** 2).sum()
    ss_tot = (w[:, None] * (beliefs_np - Y_mean) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _paper_regression_rmse(
    activations_np: np.ndarray,
    beliefs_np: np.ndarray,
    weights_np: np.ndarray,
    n_splits: int = 10,
    device: str = "cpu",
) -> float:
    """Run the paper's regression and return weighted RMSE (norm_dist)."""
    acts_t = torch.tensor(activations_np, dtype=torch.float32, device=device)
    bel_t = torch.tensor(beliefs_np, dtype=torch.float32, device=device)
    w_t = torch.tensor(weights_np, dtype=torch.float32, device=device)

    N = len(activations_np)
    indices = np.arange(N)
    fold_size = N // n_splits
    kf = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_splits - 1 else N
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        kf.append((train_idx, test_idx))

    result = run_activation_to_beliefs_regression_kf(
        regression_analyzer=None,
        activations=acts_t,
        beliefs=bel_t,
        probs=w_t,
        kf=kf,
        rcond_values=RCOND_SWEEP_LIST,
    )

    return float(result.get("norm_dist", float("inf")))
