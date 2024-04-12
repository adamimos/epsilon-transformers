import torch
from epsilon_transformers.persistence import S3Persister
from epsilon_transformers.training.configs import RawModelConfig
import wandb
import os
import shutil
import pandas as pd
from tqdm import tqdm
import csv
import tempfile
import numpy as np

def get_model_hyperparams(state_dict):
    # Vocabulary size (d_vocab)
    d_vocab = state_dict['embed.W_E'].shape[0]

    # Hidden dimension (d_model)
    d_model = state_dict['embed.W_E'].shape[1]

    # Number of attention heads (n_head)
    n_head = state_dict['blocks.0.attn.W_Q'].shape[0]

    # Head dimension (d_head)
    d_head = state_dict['blocks.0.attn.W_Q'].shape[2]

    # Number of layers (n_layers)
    n_layers = sum(1 for key in state_dict.keys() if key.startswith('blocks.') and key.endswith('.attn.W_Q'))

    # MLP dimension (d_mlp)
    d_mlp = state_dict['blocks.0.mlp.W_in'].shape[1]

    # Context size (n_ctx)
    n_ctx = state_dict['pos_embed.W_pos'].shape[0]

    hyperparams = {
        'd_vocab': d_vocab,
        'd_model': d_model,
        'n_head': n_head,
        'd_head': d_head,
        'n_layers': n_layers,
        'd_mlp': d_mlp,
        'n_ctx': n_ctx
    }

    return hyperparams

def fetch_run_config(user_or_org, project_name, run_id):
    api = wandb.Api()
    run_path = f"{user_or_org}/{project_name}/{run_id}"
    run = api.run(run_path)
    return run.config

def save_log_data_to_s3(persister, csv_file):
    persister.save_log_df(csv_file)
    os.remove(csv_file)

import torch

def load_model_artifact(
    user_or_org,
    project_name,
    artifact_name,
    artifact_type,
    artifact_version,
    config,
    device="cpu",
):
    api = wandb.Api()
    artifact_reference = f"{user_or_org}/{project_name}/{artifact_name}"
    print(f"Loading artifact {artifact_reference}")
    artifact = wandb.use_artifact(artifact_reference, type=artifact_type)

    artifact_dir = artifact.download()
    # print(f"Artifact downloaded to: {artifact_dir}")
    # get rid of the part after the : in the artifact name
    artifact_file_name = f"{artifact_name.split(':')[0]}.pt"

    # get the number, its between the final _ and the .pt
    epoch_number = artifact_file_name.split('_')[-1].split('.')[0]
    artifact_file = os.path.join(artifact_dir, artifact_file_name)

    model = build_network(config, torch.device(device))
    model.load_state_dict(torch.load(artifact_file, map_location=device))

    # delete the artifact_dir
    shutil.rmtree(artifact_dir)

    return model, epoch_number

def build_network(config, device):
    model_config = RawModelConfig(
        d_vocab=config['d_vocab'],
        d_model=config['d_model'],
        n_ctx=config['n_ctx'],
        n_head=config['n_heads'],
        d_head=config['d_head'],
        d_mlp=4*config['d_model'],
        n_layers=config['n_layers'],
    )
    return model_config.to_hooked_transformer(seed=1337, device=device)

def fetch_artifacts_for_run(user_or_org, project_name, run_id):
    api = wandb.Api()
    run_path = f"{user_or_org}/{project_name}/runs/{run_id}"
    run = api.run(run_path)
    artifacts = run.logged_artifacts()
    return artifacts

def fetch_run_history(run):
    column_names = run.history().columns
    history_data = []

    for row in tqdm(run.scan_history(page_size=50000), desc="Fetching run history"):
        history_data.append(row)

    return column_names, history_data

def create_csv_file(column_names, history_data):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='') as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()

        for row in history_data:
            writer.writerow(row)

    return file.name

def process_log_csv(csv_file):
    df = pd.read_csv(csv_file)
    # get all rows that have a value in 'val_loss'
    df_val = df[df['val_loss'].notna()]
    # get the rest
    df_train = df[df['val_loss'].isna()]

    # for each make sure we are sorting by _step
    df_val = df_val.sort_values('_step')
    df_train = df_train.sort_values('_step')

    # now add a column called train_tokens and for df_train just make that increment by 1, starting at 1
    df_train['trained_tokens'] = np.arange(1, len(df_train) + 1)*int(config['batch_size'])*int(config['n_ctx'])
    df_val['trained_tokens'] = np.arange(1, len(df_val) + 1)*int(config['n_iters'])*int(config['batch_size'])*int(config['n_ctx'])
    return df_val, df_train


if __name__ == '__main__':

    device = "cpu"
    wandb.init()
    api = wandb.Api()

    user_or_org = "adamimos"
    project_name = "transformer-MSPs"
    #run_id = '2zulyhrv' # mess3 param change
    run_id = 's6p0aaci' # zero one random
    # run_id = "halvkdvk"  # mess3 param change long run, I CANT FIND THIS ON WANDB ANYMORE
    #run_id = "vfs4q106"  # rrxor adamimos/transformer-MSPs/vfs4q106, https://wandb.ai/adamimos/transformer-MSPs/runs/vfs4q106/overview?nw=nwuseradamimos

    if run_id == "s6p0aaci":
        persister = S3Persister(collection_location='zero-one-random')
    elif run_id == "vfs4q106":
        persister = S3Persister(collection_location='rrxor')
    elif run_id == "2zulyhrv":
        persister = S3Persister(collection_location='mess3-param-change')
    else:
        raise ValueError(f"Unknown run_id: {run_id}")


    arts = fetch_artifacts_for_run(user_or_org, project_name, run_id)
    print(f"the number of artifacts is {len(arts)}")

    config = fetch_run_config(user_or_org, project_name, run_id)
    if not persister.check_if_file_exists('train_config.json'):
        persister.save_train_config(config)
    print(config)

    run_path = f"{user_or_org}/{project_name}/runs/{run_id}"
    run = api.run(run_path)

    if not persister.check_if_file_exists('train_log.csv') or not persister.check_if_file_exists('val_log.csv'):
        column_names, history_data = fetch_run_history(run)
        csv_file = create_csv_file(column_names, history_data)
        df_val, df_train = process_log_csv(csv_file)
        # save df_val and df_train
        df_val.to_csv(f"val_log.csv", index=False)
        df_train.to_csv(f"train_log.csv", index=False)
        save_log_data_to_s3(persister, f"val_log.csv")
        save_log_data_to_s3(persister, f"train_log.csv")

    # loop over artifacts
    for artifact in tqdm(arts):
        model, epoch_number = load_model_artifact(user_or_org,
                                         project_name,
                                         artifact.name,
                                         artifact.type,
                                         artifact.version,
                                         config,
                                         device)
        tokens = int(config['n_iters']) * int(config['batch_size'])* int(config['n_ctx']) * (int(epoch_number)+1)
        if not persister.check_if_file_exists(f"{tokens}.pt"):
            persister.save_model(model, tokens)
        