import fire
import pathlib
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from epsilon_transformers.persistence import Persister
from epsilon_transformers.training.configs import (
    TrainConfig,
    ProcessDatasetConfig,
    Log,
)

# TODO: Bug of outputting num_of_tokens_trained on rather than num_of_tokens_seen
# TODO: Put flag for overwriting (either don't do it, or have a logger throw a warning)

# TODO: Bug where the last final loss outputs train_loss of 0
# TODO: Use logger library for logging
# TODO: Make Log into a singleton
# TODO: Add TQDM to all of this
# TODO: Generalize train_model so that it doesn't depend on the HookedTransformer internal loss function
# TODO: move _check_if_action_batch asserts to a config validator
# TODO: Add option to resume from checkpoint

# TODO: Review best practices regarding seed setting
# TODO: Test on GPUs
# TODO: Add DP


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _calculate_tokens_trained(
    batch_size: int,
    sequence_len: int,
    batch_idx: int,
) -> int:
    tokens_per_batch = batch_size * sequence_len
    total_tokens_trained = (batch_idx + 1) * tokens_per_batch
    return total_tokens_trained


def _check_if_action_batch(
    perform_action_every_n_tokens: int,
    batch_size: int,
    sequence_len: int,
    batch_idx: int,
) -> bool:
    tokens_per_batch = batch_size * sequence_len
    assert (
        perform_action_every_n_tokens >= tokens_per_batch
    ), "perform_action_every_n_tokens must be greater than or equal to tokens_per_batch"
    perform_action_every_n_batches = perform_action_every_n_tokens // tokens_per_batch
    return (batch_idx + 1) % perform_action_every_n_batches == 0

def _evaluate_model(
    model: HookedTransformer,
    eval_dataloader: DataLoader,
    device: torch.device,
    log: Log
) -> Log:
    with torch.no_grad():
        for input_data, target_data in tqdm(eval_dataloader, desc="Eval Loop"):
            input_data, target_data = input_data.to(device), target_data.to(device)
            loss = model(input_data, return_type="loss")
            log.update_metrics("test", loss.item())
    return log


def _evaluate_log_and_persist(
    dataset_config: ProcessDatasetConfig,
    persister: Persister,
    model: HookedTransformer,
    verbose: bool,
    log: Log,
    device: torch.device,
    tokens_trained: int,
):
    eval_dataloader = dataset_config.to_dataloader(
        sequence_length=model.cfg.n_ctx, train=False
    )
    _evaluate_model(
        model=model,
        eval_dataloader=eval_dataloader,
        device=device,
        log=log
    )
    
    if verbose:
        print(f"This is the log\n{log}")
    
    log.persist()
    log.reset()
    persister.save_model(model, tokens_trained)
    return log

def train_model(config: TrainConfig) -> HookedTransformer:
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    _set_random_seed(config.seed)

    model = config.model.to_hooked_transformer(device=device, seed=config.seed)
    optimizer = config.optimizer.from_model(model=model, device=device)
    train_dataloader = config.dataset.to_dataloader(
        sequence_length=model.cfg.n_ctx, train=True
    )
    
    persister = config.persistance.init()
    log = config.init_logger()
    model.train()
    for batch_idx, (input_data, target_data) in enumerate(tqdm(train_dataloader, desc="Train Loop")):
        input_data, target_data = input_data.to(device), target_data.to(device)
        loss = model(input_data, return_type="loss")
        log.update_metrics(train_or_test="train", loss=loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tokens_trained_so_far = _calculate_tokens_trained(
            batch_size=config.dataset.batch_size,
            sequence_len=config.model.n_ctx,
            batch_idx=batch_idx,
        )

        if _check_if_action_batch(
            perform_action_every_n_tokens=config.persistance.checkpoint_every_n_tokens,
            batch_size=config.dataset.batch_size,
            batch_idx=batch_idx,
            sequence_len=config.model.n_ctx,
        ):
            model.eval()
            _evaluate_log_and_persist(
                dataset_config=config.dataset,
                persister=persister,
                model=model,
                log=log,
                verbose=config.verbose,
                device=device,
                tokens_trained=tokens_trained_so_far,
            )
            log.reset()
            model.train()

    model.eval()
    _evaluate_log_and_persist(
        dataset_config=config.dataset,
        persister=persister,
        model=model,
        log=log,
        verbose=config.verbose,
        device=device,
        tokens_trained=tokens_trained_so_far,
    )

    config.logging.close()
    return model, log


def _main(config_path: pathlib.Path):
    config: TrainConfig = TrainConfig.from_yaml(config_path)
    train_model(config)

if __name__ == "__main__":
    fire.Fire(_main)
