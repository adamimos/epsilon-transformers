import wandb
from typing import Any
import wandb
import torch
from tqdm import tqdm
import os
from multiprocessing import Pool


def fetch_artifacts_for_run(user_or_org: str, project_name: str, run_id: str) -> list:
    api = wandb.Api()
    run_path = f"{user_or_org}/{project_name}/runs/{run_id}"
    run = api.run(run_path)
    
    per_page = 1000000  # Set to a large number to fetch all artifacts in one go
    
    artifacts = run.logged_artifacts(per_page=per_page)
    
    return artifacts

def fetch_run_config(user_or_org: str, project_name: str, run_id: str) -> dict:
    api = wandb.Api()
    run_path = f"{user_or_org}/{project_name}/{run_id}"
    run = api.run(run_path)
    return run.config


def load_model_artifact(
    user_or_org: str,
    project_name: str,
    artifact_name: str,
    artifact_type: str,
    artifact_version: str,
    device: str = "cpu",
) -> Any:
    artifact_reference = (
        f"{user_or_org}/{project_name}/{artifact_name}:{artifact_version}"
    )
    print(f"Loading artifact {artifact_reference}")
    artifact = wandb.use_artifact(artifact_reference, type=artifact_type)
    artifact_dir = artifact.download()
    artifact_file = (
        f"{artifact_dir}/{artifact_name}.pt"  # Making the filename programmatic
    )
    return torch.load(artifact_file, map_location=device)
import logging
import contextlib
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

def download_artifacts(artifacts, download_dir):
    existing_pt_files = len([f for f in os.listdir(download_dir) if f.endswith('.pt')])
    print(f"Found {existing_pt_files} existing .pt files in {download_dir}")
    with tqdm(total=len(artifacts) - existing_pt_files, desc="Downloading artifacts") as pbar:
        for i, artifact in enumerate(artifacts):
            if i < existing_pt_files:
                continue
            artifact_path = os.path.join(download_dir, artifact.name)
            if not os.path.exists(artifact_path):
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    artifact.download(root=download_dir)
            pbar.set_postfix_str(f"Artifact: {artifact.name}")
            pbar.update(1)
