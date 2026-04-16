"""
Shuffle Control Analysis for Quantum RRXOR Experiments.

For each trained model, computes 6 regression curves across layers:
  1. Activations → beliefs           (main result)
  2. Activations → shuffled beliefs   (shuffle control)
  3. Next-token probs → beliefs       (NTP baseline)
  4. Beliefs → shuffled beliefs       (theoretical control)
  5. Next-token probs → shuffled      (shuffle indifference check)
  6. Random activations → beliefs     (random baseline)

Also supports running across multiple checkpoints for training dynamics.

Usage:
    uv run python scripts/activation_analysis/run_shuffle_analysis.py \
        --results_dir results/SWEEP_ID \
        --outdir shuffle_results

    # Single run, all checkpoints for training dynamics:
    uv run python scripts/activation_analysis/run_shuffle_analysis.py \
        --results_dir results/SWEEP_ID \
        --run_index 0 --all_checkpoints
"""
import argparse
import json
import os
import yaml
import joblib
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig

from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.training.dataloader import generate_all_seqs
from scripts.activation_analysis.shuffle_control import (
    compute_emission_matrix,
    compute_projections,
    geometric_shuffle,
    _paper_regression_r2,
)


DEVICE = "cuda:0"
N_SHUFFLES = 200
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------

def load_run_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run_config.yaml")) as f:
        return yaml.safe_load(f)


def list_checkpoints(run_dir: str) -> list:
    """List all checkpoint files sorted by token count."""
    pts = [f for f in os.listdir(run_dir) if f.endswith(".pt")]
    pts.sort(key=lambda f: int(f.replace(".pt", "")))
    return pts


def load_model(run_dir: str, checkpoint: str, device: str) -> HookedTransformer:
    with open(os.path.join(run_dir, "hooked_model_config.json")) as f:
        model_cfg = json.load(f)
    model_cfg["dtype"] = getattr(torch, model_cfg["dtype"].split(".")[-1])
    model_cfg["device"] = device
    hooked_config = HookedTransformerConfig(**model_cfg)
    model = HookedTransformer(hooked_config)
    model = model.to(device)
    state_dict = torch.load(os.path.join(run_dir, checkpoint),
                            map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_random_model(run_dir: str, device: str, seed: int = 999) -> HookedTransformer:
    """Load model architecture with random weights (no checkpoint)."""
    with open(os.path.join(run_dir, "hooked_model_config.json")) as f:
        model_cfg = json.load(f)
    model_cfg["dtype"] = getattr(torch, model_cfg["dtype"].split(".")[-1])
    model_cfg["device"] = device
    model_cfg["seed"] = seed
    hooked_config = HookedTransformerConfig(**model_cfg)
    model = HookedTransformer(hooked_config)
    model = model.to(device)
    model.eval()
    return model


def extract_activations(model, inputs, device):
    """Extract activations from all layers at last token position.

    Includes: resid_pre (each layer), resid_post (last layer),
    post-layernorm (ln_final), and logits.
    """
    inputs = inputs.to(device)
    n_layers = model.cfg.n_layers

    hook_names = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
    hook_names.append(f"blocks.{n_layers - 1}.hook_resid_post")
    hook_names.append("ln_final.hook_normalized")

    with torch.no_grad():
        logits, cache = model.run_with_cache(inputs, names_filter=hook_names)

    acts = {}
    for name in hook_names:
        acts[name] = cache[name][:, -1, :].cpu().numpy()

    # Logits at last position: (batch, vocab_size)
    acts["logits"] = logits[:, -1, :].cpu().numpy()

    return acts


def get_layer_order(acts_keys):
    """Return layer keys in a sensible plotting order."""
    layers = [k for k in acts_keys if k.startswith("blocks.") and "resid_pre" in k]
    layers.sort(key=lambda k: int(k.split(".")[1]))
    post = [k for k in acts_keys if "resid_post" in k and k.startswith("blocks.")]
    ln = [k for k in acts_keys if "ln_final" in k]
    logit = [k for k in acts_keys if k == "logits"]
    return layers + post + ln + logit


# ---------------------------------------------------------------------------
# Prepare process data (beliefs, next-token probs, shuffled targets)
# ---------------------------------------------------------------------------

def prepare_process_data(config, n_shuffles=1, seed=42):
    """Generate beliefs, next-token probs, and shuffled beliefs from the process."""
    process_params = config["process_config"]
    T = get_matrix_from_args(**process_params)
    ghmm = TransitionMatrixGHMM(T)
    ghmm.name = process_params["name"]
    rev = ghmm.right_eigenvector.squeeze()

    n_ctx = config["model_config"]["n_ctx"]
    seqs, probs, _ = generate_all_seqs(ghmm, n_ctx + 1, bos=False)
    tree = ghmm.derive_mixed_state_tree(depth=n_ctx + 2)

    # Belief states at the last position the MODEL sees.
    # seqs have n_ctx+1 tokens; the model input is seqs[:, :-1] (n_ctx tokens).
    # The belief after seeing n_ctx tokens is bs[-2] on the full path,
    # or equivalently bs[-1] on the truncated path.
    beliefs_list = []
    for path in [list(s.numpy()) for s in seqs]:
        bs = tree.path_to_beliefs(path[:-1])  # beliefs for the model's input
        beliefs_list.append(bs[-1].squeeze())
    beliefs = np.array(beliefs_list)
    weights = probs.numpy()

    # Next-token probabilities: p(x=0 | belief)
    # p(0|eta) = eta @ T[0] @ rev / (eta @ rev)
    p0 = (beliefs @ T[0] @ rev) / (beliefs @ rev)
    ntp = p0.reshape(-1, 1)  # (N, 1) — single feature for binary tokens

    # Emission matrix and projections
    E = compute_emission_matrix(T, rev)
    P_E, P_E_perp = compute_projections(E)

    # Generate shuffled beliefs
    rng = np.random.default_rng(seed)
    shuffled = geometric_shuffle(beliefs, P_E, P_E_perp, rng)

    return {
        "seqs": seqs,
        "beliefs": beliefs,
        "weights": weights,
        "ntp": ntp,
        "shuffled": shuffled,
        "T": T,
        "rev": rev,
        "E": E,
        "P_E": P_E,
        "P_E_perp": P_E_perp,
    }


# ---------------------------------------------------------------------------
# Compute all 6 curves for one checkpoint
# ---------------------------------------------------------------------------

def compute_curves(acts, random_acts, proc_data, device="cpu"):
    """Compute R² for all regression curves across layers.

    Uses the paper's regression procedure (SVD pseudoinverse with
    cross-validated rcond selection) for consistency.

    Returns dict: layer_name -> {curve_name: r2}
    """
    beliefs = proc_data["beliefs"]
    shuffled = proc_data["shuffled"]
    ntp = proc_data["ntp"]
    weights = proc_data["weights"]

    layer_order = get_layer_order(acts.keys())
    results = {}

    for layer in layer_order:
        act = acts[layer]
        rand_act = random_acts[layer]

        r = {}
        r["acts_to_beliefs"] = _paper_regression_r2(act, beliefs, weights, device=device)
        r["acts_to_shuffled"] = _paper_regression_r2(act, shuffled, weights, device=device)
        r["ntp_to_beliefs"] = _paper_regression_r2(ntp, beliefs, weights, device=device)
        r["beliefs_to_shuffled"] = _paper_regression_r2(beliefs, shuffled, weights, device=device)
        r["ntp_to_shuffled"] = _paper_regression_r2(ntp, shuffled, weights, device=device)
        r["random_to_beliefs"] = _paper_regression_r2(rand_act, beliefs, weights, device=device)

        results[layer] = r

    return results


# ---------------------------------------------------------------------------
# Shuffle distribution (for p-values)
# ---------------------------------------------------------------------------

def compute_shuffle_distribution(acts, proc_data, n_shuffles=N_SHUFFLES,
                                  seed=42, device="cpu"):
    """Run multiple shuffles on the final layer to get distribution of R²."""
    rng = np.random.default_rng(seed)
    beliefs = proc_data["beliefs"]
    weights = proc_data["weights"]
    P_E = proc_data["P_E"]
    P_E_perp = proc_data["P_E_perp"]

    # Use the last resid_post layer
    layer_keys = get_layer_order(acts.keys())
    last_layer = [k for k in layer_keys if "resid_post" in k][-1]
    act = acts[last_layer]

    # Original
    r2_orig = _paper_regression_r2(act, beliefs, weights, device=device)

    # Shuffled distribution
    r2_shuffled = []
    for _ in tqdm(range(n_shuffles), desc="Shuffle distribution", leave=False):
        shuf = geometric_shuffle(beliefs, P_E, P_E_perp, rng)
        r2 = _paper_regression_r2(act, shuf, weights, device=device)
        r2_shuffled.append(r2)

    return {
        "r2_original": r2_orig,
        "r2_shuffled": r2_shuffled,
        "layer": last_layer,
        "p_value": float(np.mean(np.array(r2_shuffled) >= r2_orig)),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_checkpoint(run_dir, checkpoint, proc_data, random_acts,
                       device, n_shuffles):
    """Full analysis for a single checkpoint."""
    model = load_model(run_dir, checkpoint, device)
    acts = extract_activations(model, proc_data["seqs"][:, :-1].long(), device)
    del model
    torch.cuda.empty_cache()

    reg_device = device if "cuda" in device else "cpu"
    curves = compute_curves(acts, random_acts, proc_data, device=reg_device)
    shuffle_dist = compute_shuffle_distribution(
        acts, proc_data, n_shuffles=n_shuffles, device=reg_device
    )

    tokens_seen = int(checkpoint.replace(".pt", ""))
    return {
        "checkpoint": checkpoint,
        "tokens_seen": tokens_seen,
        "curves": curves,
        "shuffle_dist": shuffle_dist,
    }


def analyze_single_run(run_dir, outdir, device=DEVICE, n_shuffles=N_SHUFFLES,
                       all_checkpoints=False):
    """Run full analysis on a single training run."""
    config = load_run_config(run_dir)
    run_name = os.path.basename(run_dir)
    print(f"\n{'='*60}")
    print(f"Analyzing: {run_name}")
    print(f"{'='*60}")

    # Prepare process data (independent of checkpoint)
    proc_data = prepare_process_data(config)
    print(f"  Process: {config['process_config']['name']}")
    print(f"  Sequences: {proc_data['seqs'].shape[0]}, "
          f"Beliefs: {proc_data['beliefs'].shape}")

    # Random baseline activations (computed once)
    print("  Computing random baseline...")
    random_model = load_random_model(run_dir, device)
    random_acts = extract_activations(
        random_model, proc_data["seqs"][:, :-1].long(), device
    )
    del random_model
    torch.cuda.empty_cache()

    # Select checkpoints
    all_ckpts = list_checkpoints(run_dir)
    if all_checkpoints:
        # Sample ~20 checkpoints evenly spaced + initial + final
        n_total = len(all_ckpts)
        if n_total <= 20:
            selected = all_ckpts
        else:
            indices = np.linspace(0, n_total - 1, 20, dtype=int)
            selected = [all_ckpts[i] for i in indices]
            # Ensure initial and final are included
            if all_ckpts[0] not in selected:
                selected = [all_ckpts[0]] + selected
            if all_ckpts[-1] not in selected:
                selected.append(all_ckpts[-1])
    else:
        # Just initial and final
        selected = [all_ckpts[0], all_ckpts[-1]]

    print(f"  Analyzing {len(selected)} checkpoints: "
          f"{selected[0]} ... {selected[-1]}")

    # Analyze each checkpoint
    checkpoint_results = []
    for ckpt in tqdm(selected, desc="Checkpoints"):
        print(f"\n  Checkpoint: {ckpt}")
        result = analyze_checkpoint(
            run_dir, ckpt, proc_data, random_acts, device, n_shuffles
        )
        checkpoint_results.append(result)

        # Print summary for this checkpoint (use last resid_post layer)
        layer_keys = list(result["curves"].keys())
        last_layer = [k for k in layer_keys if "resid_post" in k][-1]
        c = result["curves"][last_layer]
        sd = result["shuffle_dist"]
        print(f"    [{last_layer}] R² acts→beliefs: {c['acts_to_beliefs']:.4f}")
        print(f"    [{last_layer}] R² acts→shuffled: {c['acts_to_shuffled']:.4f}")
        print(f"    [{last_layer}] R² ntp→beliefs:   {c['ntp_to_beliefs']:.4f}")
        print(f"    [{last_layer}] R² random→beliefs:{c['random_to_beliefs']:.4f}")
        print(f"    shuffle p-val: {sd['p_value']:.4f}")

    # Save
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{run_name}_shuffle.joblib")
    save_data = {
        "config": config,
        "process_params": config["process_config"],
        "checkpoint_results": checkpoint_results,
        "beliefs_shape": proc_data["beliefs"].shape,
        "n_shuffles": n_shuffles,
    }
    joblib.dump(save_data, out_path)
    print(f"\n  Saved to {out_path}")

    del random_acts
    torch.cuda.empty_cache()
    return save_data


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="shuffle_results")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--n_shuffles", type=int, default=N_SHUFFLES)
    parser.add_argument("--run_index", type=int, default=None,
                        help="Only analyze this run index (0-7)")
    parser.add_argument("--all_checkpoints", action="store_true",
                        help="Analyze ~20 checkpoints for training dynamics")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ])

    if args.run_index is not None:
        run_dirs = [run_dirs[args.run_index]]

    print(f"Found {len(run_dirs)} runs | device={args.device} | "
          f"shuffles={args.n_shuffles} | all_ckpts={args.all_checkpoints}")

    for run_dir in run_dirs:
        try:
            analyze_single_run(
                str(run_dir), args.outdir,
                device=args.device,
                n_shuffles=args.n_shuffles,
                all_checkpoints=args.all_checkpoints,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
