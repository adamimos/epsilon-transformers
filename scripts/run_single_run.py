#%%
# run_single_run.py
import sys
import os
import torch
import copy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data

def run_linear_regression_with_probs(X, Y, nn_word_probs):
    reg = LinearRegression()
    reg.fit(X, Y, sample_weight=nn_word_probs)
    Y_pred = reg.predict(X)
    distances = np.sqrt(np.sum((Y_pred - Y)**2, axis=1))
    weighted_distances = distances * nn_word_probs
    return weighted_distances.sum()

def process_single_layer(act_tensor, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs):
    norm_dist = run_linear_regression_with_probs(
        act_tensor.numpy().reshape(-1, act_tensor.shape[-1]),
        nn_beliefs.numpy().reshape(-1, nn_beliefs.shape[-1]),
        nn_word_probs.numpy().reshape(-1) / nn_word_probs.numpy().reshape(-1).sum()
    )
    unnorm_dist = run_linear_regression_with_probs(
        act_tensor.numpy().reshape(-1, act_tensor.shape[-1]),
        nn_unnormalized_beliefs.numpy().reshape(-1, nn_unnormalized_beliefs.shape[-1]),
        nn_word_probs.numpy().reshape(-1) / nn_word_probs.numpy().reshape(-1).sum()
    )
    return norm_dist, unnorm_dist

def process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs):
    records = []
    for layer_idx, (layer_name, act_tensor) in enumerate(acts.items()):
        norm_dist, unnorm_dist = process_single_layer(act_tensor, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        records.append({
            "layer_name": layer_name,
            "layer_idx": layer_idx,
            "norm_dist": norm_dist,
            "unnorm_dist": unnorm_dist
        })
    concat_acts = torch.cat([acts[k] for k in acts], dim=-1)
    norm_dist, unnorm_dist = process_single_layer(concat_acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
    records.append({
        "layer_name": "concat",
        "layer_idx": layer_idx + 1,
        "norm_dist": norm_dist,
        "unnorm_dist": unnorm_dist
    })
    return pd.DataFrame(records)

def shuffle_weights(model: HookedTransformer) -> HookedTransformer:
    shuffled_model = copy.deepcopy(model)
    for name, param in shuffled_model.named_parameters():
        if param.requires_grad and param.ndim > 1:
            flat = param.data.view(-1)
            perm = torch.randperm(flat.size(0))
            shuffled_flat = flat[perm]
            param.data.copy_(shuffled_flat.view(param.data.size()))
    return shuffled_model

def extract_checkpoint_number(ckpt_path: str) -> int:
    base = os.path.basename(ckpt_path)
    num_str, _ = os.path.splitext(base)
    try:
        return int(num_str)
    except ValueError:
        return num_str

def main(run_id):
    sweep_id = '20241205175736'
    max_layers = 10
    relevant_activation_keys = (
        ['blocks.0.hook_resid_pre'] + 
        [f'blocks.{i}.hook_resid_post' for i in range(max_layers)] + 
        ['ln_final.hook_normalized']
    )
    base_outdir = f"analysis_cache_20250218/{sweep_id}"
    os.makedirs(base_outdir, exist_ok=True)

    loader = S3ModelLoader()

    # Load initial checkpoint and prepare data
    init_checkpoint = loader.list_checkpoints(sweep_id, run_id)[0]
    model, run_config = loader.load_checkpoint(sweep_id, run_id, init_checkpoint, device='cpu')
    nn_inputs, nn_beliefs, _, nn_word_probs, nn_unnormalized_beliefs = prepare_msp_data(
        run_config, run_config["model_config"]
    )
    model_cfg = model.cfg

    results = []
    for rndm_seed in tqdm(range(10), desc="Random seeds"):
        model_cfg.seed = rndm_seed
        random_model = HookedTransformer(model_cfg)
        _, acts = random_model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)
        df = process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = 'RANDOM'
        df['random_or_trained'] = 'random'
        results.append(df)

    for checkpoint in tqdm(loader.list_checkpoints(sweep_id, run_id), desc="Trained checkpoints"):
        checkpoint_number = extract_checkpoint_number(checkpoint)
        model, run_config = loader.load_checkpoint(sweep_id, run_id, checkpoint, device='cpu')
        _, acts = model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)
        df = process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = checkpoint_number
        df['random_or_trained'] = 'trained'
        results.append(df)

        shuffled_model = shuffle_weights(model)
        _, acts = shuffled_model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)
        df = process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = checkpoint_number
        df['random_or_trained'] = 'shuffled'
        results.append(df)

    results_df = pd.concat(results)
    csv_path = os.path.join(base_outdir, f"{run_id}_3.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results written to {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_single_run.py <run_id>")
        sys.exit(1)
    run_id = sys.argv[1]
    main(run_id)


