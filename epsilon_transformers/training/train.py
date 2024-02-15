import fire
import pathlib
import random
import numpy as np
import torch

from epsilon_transformers.training.configs import TrainConfig

# TODO: Add eval part of the code
# TODO: Couple eval, logging & saving frequencies (??)
# TODO: Test _check_if_action_batch()
# TODO: Review best practices regarding seed setting
# TODO: Add Wandb Logging
# TODO: Test on GPUs
# TODO: Implement to_hooked_transformer()
# TODO: Implement save_model()
# TODO: Add DP

def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _check_if_action_batch(action_frequency: int, total_tokens: int, batch_size: int, batch_idx: int) -> bool:
    total_batches = total_tokens // batch_size
    action_interval = total_batches // action_frequency
    return (batch_idx + 1) % action_interval

def _main(config_path: pathlib.Path):
    config: TrainConfig = TrainConfig.from_yaml(config_path)
    train_model(config)

def train_model(config: TrainConfig):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    _set_random_seed(config.seed)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    model = config.model_config.to_hooked_transformer(device=device)
    optimizer = config.optimizer_config.from_model(model=model, device=device)
    dataloader = config.dataset_config.to_dataloader(sequence_length=model.cfg.n_ctx, device=device)
    
    model.train()
    total_loss = 0.0
    for batch_idx, (input_data, target_data) in enumerate(dataloader):
        input_data, target_data = input_data.to(model.device), target_data.to(model.device)

        predictions = model(input_data)
        loss = criterion(predictions, target_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # TODO: Logging
        
        if _check_if_action_batch(action_frequency=config.persistance_config.checkpoint_every_n_tokens, total_tokens=config.dataset_config.num_tokens, batch_size=config.dataset_config.batch_size, batch_idx=batch_idx):
            config.persistance_config.save_model()


if __name__ == "__main__":
    fire.Fire(_main)