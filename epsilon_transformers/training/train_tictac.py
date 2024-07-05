import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
from epsilon_transformers.process.tictac import create_game_dataframe, get_full_sequences, compute_block_entropy, compute_myopic_entropy
from epsilon_transformers.training.configs.training_configs import TrainConfig
from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.training.configs.training_configs import OptimizerConfig, PersistanceConfig, LoggingConfig, ProcessDatasetConfig

def train_epoch(model, optimizer, dataset):
    model.train()

    loss_agg = []

    for batch, weights in tqdm(dataset):
        optimizer.zero_grad(set_to_none=True)
        batch, weights = batch.cuda(), weights.cuda()
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, model.cfg.d_vocab),
            batch[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(batch.shape[0], -1)

        weighted_loss = loss * weights.reshape(-1, 1)
        weighted_loss.sum().backward()
        loss_agg.append(weighted_loss)

        optimizer.step()
    return torch.concat(loss_agg).sum(axis=0).detach().cpu().numpy()

def train_model(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate data from tictac
    game_df = create_game_dataframe()
    sequences, probs = get_full_sequences(game_df, pad_token=9, bos=9)
    sequences = torch.tensor(sequences, dtype=torch.long).to(device)
    probs = torch.tensor(probs, dtype=torch.float32).to(device)

    batch_size = config.dataset.batch_size

    block_entropy = compute_block_entropy(game_df)
    myopic_entropy = compute_myopic_entropy(block_entropy)

    model = config.model.to_hooked_transformer(device=device.type, seed=config.seed)
    optimizer = config.optimizer.from_model(model=model, device=device)
    
    persister = config.persistance.init()
    persister.save_config(config)

    save_every = 10

    for epoch in range(20000):
        indices = torch.randperm(len(sequences))
        dataset = [
            (sequences[indices[i:i+batch_size]], probs[indices[i:i+batch_size]])
            for i in range(0, len(sequences), batch_size)
        ]
        loss = train_epoch(model, optimizer, dataset)
        
        rel_loss = myopic_entropy / loss

        torch.set_printoptions(precision=5, sci_mode=False, linewidth=1000)
        with np.printoptions(precision=5, suppress=True, linewidth=1000):
            print(f"Epoch {epoch}\nLoss: {loss}\nRelative Loss: {rel_loss}")
        

        with open(f"{persister.collection_location}/loss_log_{epoch}.pt", "wb") as f:
            torch.save({
                "loss": loss,
                "rel_loss": rel_loss
            }, f)

        if epoch % save_every == 0:
            persister.save_model(model, epoch * len(sequences))

if __name__ == "__main__":
    dm = 16
    model_config = RawModelConfig(
        d_vocab=10,  # 0-8 for moves, 9 for padding
        d_model=dm,
        n_ctx=10,
        d_head=dm,
        n_head=1,
        d_mlp=dm * 4,
        n_layers=1,
        attn_only=True,
    )

    optimizer_config = OptimizerConfig(
        optimizer_type="adam",
        learning_rate=1e-3,
        weight_decay=0,
    )

    lp = "/home/ubuntu/mats-1"

    from datetime import datetime
    now = datetime.now()
    folder = f'tictac_runs/{now.strftime("%Y%m%d-%H%M%S")}-tictac'
    from pathlib import Path

    p = Path(lp) / folder
    p.mkdir(parents=True, exist_ok=True)

    persistance_config = PersistanceConfig(
        location="local",
        collection_location=p,
        checkpoint_every_n_tokens=10000
    )


    dataset = ProcessDatasetConfig(
        batch_size=512,
        num_tokens=1000000,
        test_split=0.1,
        process="tictac",
        process_params={},
    )

    config = TrainConfig(
        model=model_config,
        optimizer=optimizer_config,
        persistance=persistance_config,
        logging=LoggingConfig(project_name="tictac-training", wandb=False),
        dataset=dataset,
        verbose=True,
        seed=42,
    )

    train_model(config)