"""
Shuffle Control Analysis for Quantum RRXOR Experiments.

Loads trained transformer checkpoints, extracts activations, computes
ground-truth belief states, and runs the geometric shuffle control to test
whether the network encodes belief structure beyond next-token prediction.

Analogous to run_regression_analysis.py but for the shuffle control experiment.

Usage:
    uv run python scripts/activation_analysis/run_shuffle_analysis.py \
        --results_dir results/SWEEP_ID \
        --outdir shuffle_results
"""
import argparse
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
    run_shuffle_control,
    run_perpendicular_regression,
)


DEVICE = "cuda:0"
N_SHUFFLES = 200


def load_run_config(run_dir: str) -> dict:
    """Load the run config from a training run directory."""
    config_path = os.path.join(run_dir, "run_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_final_checkpoint(run_dir: str) -> str:
    """Find the final (highest token count) checkpoint file."""
    checkpoints = [f for f in os.listdir(run_dir) if f.endswith(".pt") and f != "0.pt"]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {run_dir}")
    # Sort by token count (filename is the token count)
    checkpoints.sort(key=lambda f: int(f.replace(".pt", "")))
    return checkpoints[-1]


def load_model(run_dir: str, checkpoint: str, device: str) -> HookedTransformer:
    """Load a trained HookedTransformer from a checkpoint."""
    # Load model config
    config_path = os.path.join(run_dir, "hooked_model_config.json")
    import json
    with open(config_path) as f:
        model_cfg = json.load(f)

    # Fix dtype
    model_cfg["dtype"] = getattr(torch, model_cfg["dtype"].split(".")[-1])
    model_cfg["device"] = device

    hooked_config = HookedTransformerConfig(**model_cfg)
    model = HookedTransformer(hooked_config)
    model = model.to(device)

    # Load weights
    state_dict = torch.load(os.path.join(run_dir, checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_activations(
    model: HookedTransformer,
    inputs: torch.Tensor,
    device: str,
) -> dict:
    """Extract residual stream activations from all layers at the last token position.

    Returns dict mapping layer name -> activations array (N, d_model).
    """
    inputs = inputs.to(device)
    n_layers = model.cfg.n_layers

    # Hook names for residual stream pre-attention at each layer + final
    hook_names = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
    hook_names.append(f"blocks.{n_layers - 1}.hook_resid_post")

    with torch.no_grad():
        _, cache = model.run_with_cache(inputs, names_filter=hook_names)

    acts = {}
    for name in hook_names:
        # Take last token position: (batch, seq_len, d_model) -> (batch, d_model)
        a = cache[name][:, -1, :].cpu().numpy()
        acts[name] = a

    # Combined: concatenation of all layers
    combined = np.concatenate([acts[name] for name in hook_names], axis=1)
    acts["combined"] = combined

    return acts


def analyze_single_run(
    run_dir: str,
    outdir: str,
    device: str = DEVICE,
    n_shuffles: int = N_SHUFFLES,
):
    """Run shuffle control analysis on a single training run."""
    config = load_run_config(run_dir)
    run_name = os.path.basename(run_dir)
    print(f"\n{'='*60}")
    print(f"Analyzing: {run_name}")
    print(f"{'='*60}")

    # Load process
    process_params = config["process_config"]
    T = get_matrix_from_args(**process_params)
    ghmm = TransitionMatrixGHMM(T)
    ghmm.name = process_params["name"]
    rev = ghmm.right_eigenvector.squeeze()

    print(f"  Process: {process_params['name']}")
    print(f"  T shape: {T.shape}, rev shape: {rev.shape}")

    # Generate all sequences and belief states
    n_ctx = config["model_config"]["n_ctx"]
    seqs, probs, _ = generate_all_seqs(ghmm, n_ctx + 1, bos=False)
    tree = ghmm.derive_mixed_state_tree(depth=n_ctx + 2)

    # Get belief states for each sequence at the last position
    beliefs_list = []
    for path in [list(s.numpy()) for s in seqs]:
        bs = tree.path_to_beliefs(path)
        beliefs_list.append(bs[-1].squeeze())  # last position belief
    beliefs = np.array(beliefs_list)
    weights = probs.numpy()

    print(f"  Sequences: {seqs.shape[0]}, Beliefs: {beliefs.shape}")

    # Load model
    checkpoint = get_final_checkpoint(run_dir)
    print(f"  Loading checkpoint: {checkpoint}")
    model = load_model(run_dir, checkpoint, device)

    # Extract activations
    print("  Extracting activations...")
    acts = extract_activations(model, seqs[:, :-1].long(), device)

    # Run shuffle control on each layer + combined
    results = {}
    for layer_name, act_array in acts.items():
        print(f"  Shuffle control on {layer_name}...")
        sc = run_shuffle_control(
            act_array, beliefs, T, rev,
            weights=weights, n_shuffles=n_shuffles, seed=42,
            show_progress=True,
        )
        results[layer_name] = sc
        print(f"    RMSE orig={sc['mse_original']:.6f}, "
              f"shuffled={sc['mse_shuffle_mean']:.6f} "
              f"(effect={sc['effect_size']:.1%}, p={sc['p_value']:.4f})")

    # Also run perpendicular regression on combined
    print("  Perpendicular regression on combined...")
    perp = run_perpendicular_regression(
        acts["combined"], beliefs, T, rev,
        weights=weights, n_shuffles=n_shuffles, seed=42,
    )
    results["perp_combined"] = perp
    print(f"    RMSE perp orig={perp['mse_perp_original']:.6f}, "
          f"shuffled={np.mean(perp['mse_perp_shuffled']):.6f} "
          f"(effect={perp['effect_size']:.1%}, p={perp['p_value']:.4f})")

    # Save results
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{run_name}_shuffle.joblib")
    save_data = {
        "results": results,
        "config": config,
        "process_params": process_params,
        "beliefs_shape": beliefs.shape,
        "n_shuffles": n_shuffles,
        "checkpoint": checkpoint,
    }
    joblib.dump(save_data, out_path)
    print(f"  Saved to {out_path}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to sweep results directory (e.g., results/20260416001120)")
    parser.add_argument("--outdir", type=str, default="shuffle_results",
                        help="Output directory for shuffle analysis results")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="CUDA device to use")
    parser.add_argument("--n_shuffles", type=int, default=N_SHUFFLES,
                        help="Number of shuffle permutations")
    parser.add_argument("--run_index", type=int, default=None,
                        help="Only analyze this run index (0-7). Default: all runs.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ])

    if args.run_index is not None:
        run_dirs = [run_dirs[args.run_index]]

    print(f"Found {len(run_dirs)} run directories")
    print(f"Output: {args.outdir}")
    print(f"Device: {args.device}")
    print(f"Shuffles: {args.n_shuffles}")

    all_results = {}
    for run_dir in run_dirs:
        try:
            r = analyze_single_run(
                str(run_dir), args.outdir,
                device=args.device, n_shuffles=args.n_shuffles,
            )
            all_results[run_dir.name] = r
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Run':>45s}  {'RMSE orig':>10s}  {'RMSE shuf':>10s}  {'Effect':>8s}  {'p':>6s}")
    print("-" * 85)
    for name, results in all_results.items():
        if "combined" in results:
            sc = results["combined"]
            print(f"{name:>45s}  {sc['mse_original']:10.6f}  "
                  f"{sc['mse_shuffle_mean']:10.6f}  "
                  f"{sc['effect_size']:7.1%}  {sc['p_value']:6.4f}")


if __name__ == "__main__":
    main()
