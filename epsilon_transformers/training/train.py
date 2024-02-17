import fire
import pathlib
import random
import numpy as np
import torch
from transformer_lens import HookedTransformer

from epsilon_transformers.training.configs import TrainConfig, ProcessDatasetConfig, PersistanceConfig, LoggingConfig, Log

# TODO: Add TQDM to all of this
# TODO: Generalize train_model so that it doesn't depend on the HookedTransformer internal loss function
# TODO: move _check_if_action_batch asserts to a config validator
# TODO: Add option to resume from checkpoint

# TODO: I don't like how the log is mutable... think on whether there's something better you can do

# TODO: Review best practices regarding seed setting
# TODO: Add Wandb Logging
# TODO: Test on GPUs
# TODO: Add DP

def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _check_if_action_batch(perform_action_every_n_tokens: int, batch_size: int, sequence_len: int, batch_idx: int) -> bool:
    tokens_per_batch = (batch_size * sequence_len)
    assert perform_action_every_n_tokens >= tokens_per_batch, "perform_action_every_n_tokens must be greater than or equal to tokens_per_batch"
    perform_action_every_n_batches = perform_action_every_n_tokens // tokens_per_batch
    return (batch_idx + 1) % perform_action_every_n_batches == 0

def _evaluate_model(model: HookedTransformer, eval_dataloader, eval_config, logging_config):
    with torch.no_grad():
        raise NotImplementedError

def _evaluate_log_and_persist(dataset_config: ProcessDatasetConfig, logging_config: LoggingConfig, persistance_config: PersistanceConfig, model: HookedTransformer, log: Log):
    eval_dataloader = dataset_config.to_dataloader(sequence_length=model.cfg.n_ctx, train=False)
    eval_log = _evaluate_model(model, eval_dataloader, logging_config)
    updated_log = log.merge_mutually_exclusive_logs(eval_log)
    logging_config.log(updated_log)
    persistance_config.save_model(model, dataset_config.num_tokens)
    return updated_log

def _main(config_path: pathlib.Path):
    config: TrainConfig = TrainConfig.from_yaml(config_path)
    train_model(config)

def train_model(config: TrainConfig) -> HookedTransformer:
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    _set_random_seed(config.seed)

    model = config.model.to_hooked_transformer(device=device, seed=config.seed)
    optimizer = config.optimizer.from_model(model=model, device=device)
    train_dataloader = config.dataset.to_dataloader(sequence_length=model.cfg.n_ctx, train=True)

    log = config.init_logger()

    model.train()
    for batch_idx, (input_data, target_data) in enumerate(train_dataloader):
        input_data, target_data = input_data.to(device), target_data.to(device)

        loss = model(input_data, return_type="loss")
        log = config.logging.update_train_metrics(log, loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if _check_if_action_batch(perform_action_every_n_tokens=config.persistance.checkpoint_every_n_tokens, batch_size=config.dataset.batch_size, batch_idx=batch_idx, sequence_len=config.model.n_ctx):
            model.eval()
            _evaluate_log_and_persist(dataset_config=config.dataset, logging_config={}, persistance_config=config.persistance, model=model)
            log = config.logging.init() # TODO: Check if this is actually what should be happening
            model.train()
  
    model.eval()
    final_log = _evaluate_log_and_persist(dataset_config=config.dataset, logging_config={}, persistance_config=config.persistance, model=model, final=True)
    config.logging.close()
    return model, final_log

if __name__ == "__main__":
    fire.Fire(_main)