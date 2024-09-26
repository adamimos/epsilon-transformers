import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method('spawn')

import os
import time
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
import pandas as pd
from epsilon_transformers.persistence import HackyPersister
from epsilon_transformers.training.configs.training_configs import TrainConfig
from epsilon_transformers.training.eval_functions import plot_data, run_model_eval
from epsilon_transformers.training.train import _set_random_seed
import numpy as np

import pathlib
from tqdm import tqdm

import wandb


def train_epoch(model, optimizer, dataset):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    epoch_losses = []  # List to store losses for each batch

    for batch in dataset:
        # batch shape: (batch_size, sequence_length)
        # batch[:, :-1] shape: (batch_size, sequence_length - 1)
        input_sequences = batch[:, :-1]
        
        # logits shape: (batch_size, sequence_length - 1, vocab_size)
        logits = model(input_sequences)
        
        # Reshape logits and target for cross_entropy
        # reshaped_logits shape: (batch_size * (sequence_length - 1), vocab_size)
        reshaped_logits = logits.reshape(-1, model.cfg.d_vocab)
        # reshaped_targets shape: (batch_size * (sequence_length - 1))
        reshaped_targets = batch[:, 1:].reshape(-1)
        
        # Compute loss
        # loss shape: (batch_size, sequence_length - 1)
        loss = F.cross_entropy(
            reshaped_logits,
            reshaped_targets,
            reduction="none"
        ).reshape(batch.shape[0], -1)

        # Note: Weighting is commented out, but could be added here if needed
        # weighted_loss = loss * weights.reshape(-1, 1)
        batch_loss = loss  # Renamed from weighted_loss for clarity
        
        # Compute mean loss and backpropagate
        batch_loss.mean().backward()
        
        # Store the loss for this batch
        epoch_losses.append(batch_loss)

    # Perform optimization step
    optimizer.step()

    # Compute and return the mean loss across all batches
    # Shape of returned array: (sequence_length - 1,)
    return torch.concat(epoch_losses).mean(axis=0).detach().cpu().numpy()


def evaluate(model, sequences, belief_states):
    model.eval()
    with torch.no_grad():
        pass
        # logits, cache = model.run_with_cache(sequences,
        # loss = (loss * belief_states).sum(axis=0).mean()
        # TODO - measure KL divergence with belief states
    # return loss.item()


def train_model(config: TrainConfig) -> "HookedTransformer":
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    torch.set_float32_matmul_precision('high')

    _set_random_seed(config.seed)  # rename

    msp_tree = config.dataset.process_msp(10)
    eval_probs, beliefs, eval_sequences = msp_tree.collect_paths_with_beliefs(
        max_depth=9
    )

    myopic_entropy = msp_tree.myopic_entropy[1:]

    writer = CSVLogger(
        os.path.join(config.persistance.collection_location, "train_log.csv")
    )

    # writer.log_valid(0, relative_loss, relative_loss, kl_div, figure=figure)
# print path
    print(config.persistance.collection_location)


    model = config.model.to_hooked_transformer(device=device.type, seed=config.seed)
    optimizer = config.optimizer.from_model(model=model, device=device)
    
    # Add ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, cooldown=500, threshold=1e-6,
        verbose=True
    )


    persister = config.persistance.init()
    persister.save_config(config)

    save_every = 50  # TODO

    step_fn = train_epoch
    # step_fn = torch.compile(train_epoch)
    step_losses = []
    for epoch in range(1001):
        st = time.monotonic_ns()


        dataset = config.dataset.to_dataloader_gpu(config.model.n_ctx, True)
        loss = None
        sl = []
        for batch in tqdm(dataset):
            step_loss = step_fn(model, optimizer, [batch])
            sl.append(step_loss)
            
        

        loss = np.stack(sl).mean(axis=0)
        time_ms = (time.monotonic_ns() - st) / 1e6

        rel_loss = myopic_entropy / loss[:len(myopic_entropy)]

        writer.log_train(epoch, loss, rel_loss, iter_time_ms=time_ms, lr=optimizer.param_groups[0]['lr'])

        # Step the scheduler
        scheduler.step(loss.mean())

        if not epoch % save_every and epoch != 0:
            relative_loss, kl_div, belief_predictions = run_model_eval(
                model, eval_probs, beliefs, eval_sequences, myopic_entropy[:-1]
            )
            if config.dataset.process == "mess3":
                figure = plot_data(belief_predictions.reshape(-1, 3))
            else:
                figure = None

            writer.log_valid(epoch, loss, relative_loss, kl_div, figure=figure)

            persister.save_model(model, epoch * config.dataset.batch_size * config.dataset.num_tokens)

        # evaluate(model, sequences, beliefs)

class CSVLogger:
    def __init__(self, path):
        self.path = path

    def log_train(self, epoch, loss, rel_loss, **kwargs):
        row = {
            "epoch": epoch,
            "loss_mean": loss.mean(),
            "rel_loss_mean": rel_loss.mean(),
            **dict(zip([f"loss_pos_{i}" for i in range(len(loss))], loss)),
            **dict(zip([f"rel_loss_pos_{i}" for i in range(len(rel_loss))], rel_loss)),
            **kwargs,
        }

        wandb.log(row)

        print(
            f"train: epoch={epoch} loss={loss.mean():.20f} rel_loss={rel_loss.mean():.20f}"
        )

        pd.DataFrame([row]).to_csv(
            self.path,
            mode="a",
            header=not os.path.exists(self.path),
            index=False,
        )

    def log_valid(self, epoch, loss, rel_loss, kl_div, figure=None):
        row = {
            "epoch": epoch,
            "val_loss_mean": loss.mean(),
            "val_rel_loss_mean": rel_loss.mean(),
            "val_kl_div": kl_div,
            **dict(zip([f"val_loss_pos_{i}" for i in range(len(loss))], loss)),
            **dict(zip([f"val_rel_loss_pos_{i}" for i in range(len(rel_loss))], rel_loss)),
        }

        wandb.log(row)

        print(f"valid: epoch={epoch} val_loss={loss.mean():.20f} val_rel_loss={rel_loss.mean():.20f} val_kl_div={kl_div:.20f}")

        pd.DataFrame([row]).to_csv(
            self.path.replace("train_log.csv", "val_log.csv"),
            mode="a",
            header=not os.path.exists(
                self.path.replace("train_log.csv", "val_log.csv")
            ),
            index=False,
        )

        if figure is None:
            return

        # Save figure locally as PNG and log to wandb
        figure_path = self.path.replace(
            "train_log.csv", f"val_figure_epoch_{epoch}.png"
        )
        figure.savefig(figure_path, format="png", dpi=300, bbox_inches="tight")

        # Log figure to wandb
        wandb.log({"validation_figure": wandb.Image(figure)})

        # Close the figure to free up memory
        plt.close(figure)


from epsilon_transformers.training.configs.model_configs import LSTMModelConfig, RawModelConfig
from epsilon_transformers.training.configs.training_configs import (
    LoggingConfig,
    OptimizerConfig,
    PersistanceConfig,
    ProcessDatasetConfig,
    TrainConfig,
)



print("CONFIG") 
# print(model_config)
optim = "adamw"

optimizer_config = OptimizerConfig(
    optimizer_type=optim,
    learning_rate=1e-4,
    weight_decay=0,
)
# pR1=0.5, pR2=0.5):
pp = {'pR1': 0.5, 'pR2': 0.5}
pp = {"x": 0.15, "a": 0.6}
# pp = {"x": 0.05, "a": 0.85}
# pp = {"x": 0.5, "a": 0.6}
dataset_config = ProcessDatasetConfig(
    process="mess3",
    # process="rrxor",
    process_params=pp,
    batch_size=512,
    num_tokens=512 * 32 * 4,
    test_split=0.00000000015,
)

dm = 64


model_config = RawModelConfig(
    d_vocab=dataset_config.n_symbols(),
    d_model=dm,
    n_ctx=10,
    d_head=dm,
    n_head=1,
    d_mlp=dm * 4,
    n_layers=1,
    attn_only=False,
)

# model_config = LSTMModelConfig(
#     model_type="lstm",
#     d_vocab=dataset_config.n_symbols(),
#     d_model=dm,
#     n_ctx=128,
#     n_layers=1,
#     # d_head=dm,
#     # n_head=1,
#     # d_mlp=dm * 4,
#     # attn_only=False,
# )


from epsilon_transformers.server.registry import run_dir
lp = run_dir


from datetime import datetime

now = datetime.now()
# folder = f'gen5_runs/{now.strftime("%Y%m%d-%H%M%S")}-mess3-{optim}'
# folder = f'gen5_runs/{now.strftime("%Y%m%d-%H%M%S")}-mess3-x015-a085-{optim}'
# folder = f'gen7_runs/{now.strftime("%Y%m%d-%H%M%S")}-gd1-mess3-x{int(pp["x"]*100):02d}-a{int(pp["a"]*100):02d}-{optim}-'
from pathlib import Path
folder = ''
p = Path(lp) / folder
# p.mkdir(parents=True, exist_ok=True)

persistance_config = PersistanceConfig(
    location="local", collection_location=p, checkpoint_every_n_tokens=1000000
)

from random import randint
mock_config = TrainConfig(
    model=model_config,
    optimizer=optimizer_config,
    dataset=dataset_config,
    persistance=persistance_config,
    logging=LoggingConfig(project_name="mess3-gd", wandb=False),
    verbose=True,
    seed=randint(0, 1000000),
)



def create_run_folder(config: TrainConfig) -> Path:
    """
    Create a run folder based on the configuration.
    
    Args:
        config (TrainConfig): The configuration object.
    
    Returns:
        Path: The path to the created run folder.
    """
    now = datetime.now()
    process_params = '_'.join([f"{k}_{round(v*100)}" for k, v in config.dataset.process_params.items()])
    import random
    import string
    folder_name = (
        f'gen14_runs/{now.strftime("%Y%m%d-%H%M%S")}-'
        f'gd1-'
        f'{"".join(random.choices(string.ascii_lowercase + string.digits, k=6))}-'
        f'{config.dataset.process}-'
        f'{process_params}-'
        f'{config.model.model_type}-'
        f'nl{config.model.n_layers}-'
        f'lr_{config.optimizer.learning_rate:.2e}_'
        f'bs_{config.dataset.batch_size}-'
        f'{config.optimizer.optimizer_type}'
    )
    
    folder_path = Path(lp) / folder_name
    return folder_path

def handle_queue(queue):
    while not queue.empty():
        pp = queue.get()
        run_experiment(1e-4, 128, pp)

def run_experiments_parallel(pp_values):
    import signal
    queue = mp.Queue()
    max_processes = min(5, mp.cpu_count())
    for pp in pp_values:
        queue.put(pp)
    
    processes = [mp.Process(target=handle_queue, args=(queue,)) for _ in range(max_processes)]
    for p in processes:
        p.start()

    def handle_sigint(signum, frame):
        print("SIGINT received, terminating processes...")
        for p in processes:
            p.terminate()

    signal.signal(signal.SIGINT, handle_sigint)

    for p in processes:
        p.join()




def run_experiment(lr, bs, pp):
    mock_config.optimizer.learning_rate = lr
    mock_config.dataset.batch_size = bs
    mock_config.dataset.process_params = pp
    # update per dir
    # mock_config.persistance.collection_location = Path(str(p).replace('mess3-', f'mess3-nl{mock_config.model.n_layers}-lr_{lr:.2e}_bs_{bs}-'))
    mock_config.persistance.collection_location = create_run_folder(mock_config)
    mock_config.persistance.collection_location.mkdir(parents=True, exist_ok=True)

    with wandb.init(project="mess3-gd-res", config=mock_config.model_dump()):
        train_model(mock_config)

if __name__ == '__main__':
        # lr_values = [1e-2, 5e-3, 1e-3, 1e-4]
        lr_values = [1e-3, 1e-4]
        # bs_values = [64, 128, 512, 1024]
        bs_values = [64, 128, 512]

        # experiments = [(lr, bs) for lr in lr_values for bs in bs_values]

        # with mp.Pool(processes=min(4, mp.cpu_count())) as pool:
            # pool.starmap(run_experiment, experiments)
        # missing = [(0.15, 0.2), (0.1, 0.2), (0.15, 0.6), (0.1, 0.6)]
        # missing = [(0.15, 0.2)] * 4
        missing = [(0.15, .05), (0.15, .1), (0.15, .15), (0.15, .25), (0.15, .45), (0.15, .5), (0.15, .55), (0.15, .65), (0.15, .7), (0.15, .75), (0.15, .8), (0.15, .85), (0.15, .9), (0.15, .95)]
        pp_values = [{"x": x, "a": a} for x, a in missing]
        run_experiments_parallel(pp_values)

        pp_values = [{"x": .1, "a": a} for x, a in missing]
        run_experiments_parallel(pp_values)

        pp_values = [{"x": .05, "a": a} for x, a in missing]
        run_experiments_parallel(pp_values)
    
        # run_experiment(1e-4, 128 * 2, {"x": 0.15, "a": 0.6})