# type: ignore
# %%
from typing import List, Dict, Optional, Tuple
import yaml
from pydantic import BaseModel
from epsilon_transformers.comp_mech.processes import (
    random_random_xor,
    zero_one_random,
    mess3,
    serpinski
)
from epsilon_transformers.configs import SweepConfig
import torch
import torch.nn.functional as F
import wandb
import torch.nn as nn

from epsilon_transformers import (
    build_dataset,
    build_optimizer,
    build_network,
    create_validation_set,
    build_probabilistic_dataset,
)
from epsilon_transformers.comp_mech import (
    generate_sequences,
    mixed_state_tree,
    block_entropy,
    myopic_entropy,
)

# %%

with open(
    "./experiments/serpinski_sweep/serpinski_sweep_cfg.yaml", "r"
) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

try:
    validated_config = SweepConfig(**config)
    print(f"Validated config")
except Exception as e:
    print(f"Invalid config: {e}")

if config["process"] == "RRXOR":
    process = random_random_xor()
elif config["process"] == "mess3":
    process = mess3()
elif config["process"] == "zero_one_random":
    process = zero_one_random()
elif config["process"] == "serpinski":
    process = serpinski()

print(config["parameters"]["n_ctx"])
MSP_tree = mixed_state_tree(process, config["parameters"]["n_ctx"]["value"] + 1)
myopic_entropy_rate = myopic_entropy(MSP_tree)
minimum_cross_entropy = myopic_entropy_rate[1:]
print(f"myopic_entropy_rate: {myopic_entropy_rate}")

# %%
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
#device = torch.device("cpu")
print(f"Using device: {device}")

minimum_cross_entropy = torch.tensor(minimum_cross_entropy, dtype=torch.float32).to(
    device
)


# %%
def sweep_train(config: Optional[Dict] = None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        sweep_name = wandb.run.sweep_id

        # Build train loder
        # data = build_dataset(config, process)

        X_val, Y_val, val_weights = create_validation_set(MSP_tree, config.n_ctx)
        X_val = torch.tensor(X_val, dtype=torch.int).to(device)
        val_weights = torch.tensor(val_weights, dtype=torch.float32).to(device)
        Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)

        # Build model
        model = build_network(config, device)

        # Move model to device
        model.to(device)
        val_data = torch.tensor(X_val, dtype=torch.int).to(device)
        val_weights = torch.tensor(val_weights, dtype=torch.float32).to(device)

        # Build optimizer
        optimizer = build_optimizer(model, config)

        # cross entropy loss
        criterion = nn.CrossEntropyLoss(reduction="none")

        # Train model
        for epoch in range(config.num_epochs):
            train_weights = build_probabilistic_dataset(
                val_weights.cpu().numpy(), config.batch_size, config.n_iters
            )
            train_weights = torch.tensor(train_weights, dtype=torch.float32).to(device)
            train_epoch_prob(
                model,
                optimizer,
                val_data,
                Y_val,
                val_weights,
                train_weights,
                criterion,
                minimum_cross_entropy,
                epoch,
            )


def train_epoch_prob(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    val_data: torch.Tensor,
    val_output: torch.Tensor,
    val_weights: torch.Tensor,
    train_weights: torch.Tensor,
    criterion: torch.nn.Module,
    minimum_cross_entropy: torch.Tensor,
    epcoh_ind: int,
):
    model.train()
    for weights in train_weights:
        non_zero_indices = torch.nonzero(weights).squeeze()
        non_zero_weights = weights[non_zero_indices]
        # get rid of val_data that has no weight
        val_data_ = val_data[non_zero_indices]
        val_output_ = val_output[non_zero_indices]

        optimizer.zero_grad()
        Y = model(val_data_)  # Forward pass
        loss = criterion(Y.view(-1, model.cfg.d_vocab), val_output_.view(-1))
        loss = loss.view(val_data_.shape[0], val_data_.shape[1])  # *(batch_size, n_ctx)

        mean_loss, relative_loss = compute_val_losses(
            loss, minimum_cross_entropy, non_zero_weights
        )

        mean_loss.backward()
        optimizer.step()
        log_data = {
            "loss": mean_loss.item(),
            "relative_loss": relative_loss.mean().item(),
        }
        for i, rel_loss in enumerate(relative_loss):
            log_data[f"relative_loss_{i}"] = rel_loss.item()
        wandb.log(log_data)

    # validation
    model.eval()
    with torch.no_grad():
        
        # run the whole validation set
        Y = model(val_data)
        loss = criterion(Y.view(-1, model.cfg.d_vocab), val_output.view(-1))
        loss = loss.view(val_data.shape[0], val_data.shape[1])
        mean_loss, relative_loss = compute_val_losses(
            loss, minimum_cross_entropy, val_weights
        )
        log_data = {
            "val_loss": mean_loss.item(),
            "val_relative_loss": relative_loss.mean().item(),
        }
        for i, rel_loss in enumerate(relative_loss):
            log_data[f"val_relative_loss_{i}"] = rel_loss.item()
        wandb.log(log_data)

    model_state_dict = model.state_dict()
    artifact = wandb.Artifact(f"model_epoch_{epcoh_ind}", type="model")
    with artifact.new_file(f"model_epoch_{epcoh_ind}.pt", mode="wb") as fa:
        torch.save(model_state_dict, fa)
    wandb.log_artifact(artifact)


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    val_data: torch.Tensor,
    val_weights: torch.Tensor,
    minimum_cross_entropy: torch.Tensor,
):
    model.train()
    for X, Y_true in data_loader:
        X, Y_true = X.to(device), Y_true.to(device)
        optimizer.zero_grad()
        Y = model(X)  # Forward pass
        loss = criterion(Y.view(-1, model.cfg.d_vocab), Y_true.view(-1))
        loss = loss.view(X.shape[0], X.shape[1])  # *(batch_size, n_ctx)
        mean_loss, relative_loss = compute_losses(loss, minimum_cross_entropy)
        mean_loss.backward()
        optimizer.step()
        log_data = {
            "loss": mean_loss.item(),
            "relative_loss": relative_loss.mean().item(),
        }
        for i, rel_loss in enumerate(relative_loss):
            log_data[f"relative_loss_{i}"] = rel_loss.item()
        wandb.log(log_data)

    # validation
    model.eval()
    with torch.no_grad():
        Y = model(val_data)
        loss = criterion(Y.view(-1, model.cfg.d_vocab), val_data.view(-1))
        loss = loss.view(val_data.shape[0], val_data.shape[1])
        mean_loss, relative_loss = compute_val_losses(
            loss, minimum_cross_entropy, val_weights
        )
        log_data = {
            "val_loss": mean_loss.item(),
            "val_relative_loss": relative_loss.mean().item(),
        }
        for i, rel_loss in enumerate(relative_loss):
            log_data[f"val_relative_loss_{i}"] = rel_loss.item()
        wandb.log(log_data)


def compute_losses(
    loss: torch.Tensor, minimum_cross_entropy: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    per_position_loss = loss.mean(dim=0)  # *(n_ctx,)
    relative_loss = per_position_loss / minimum_cross_entropy
    mean_loss = per_position_loss.mean()
    return mean_loss, relative_loss


def compute_val_losses(
    loss: torch.Tensor, minimum_cross_entropy: torch.Tensor, val_weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    per_position_loss = torch.einsum("bp,b->p", loss, val_weights)
    relative_loss = per_position_loss / minimum_cross_entropy
    mean_loss = per_position_loss.mean()
    return mean_loss, relative_loss


sweep_id = wandb.sweep(config, project=config["sweep_name"])  # type: ignore
wandb.agent(sweep_id, function=sweep_train, count=100)  # type: ignore

# %%
