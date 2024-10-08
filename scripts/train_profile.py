import argparse
from epsilon_transformers.training.logger import StructuredLogger
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.training.dataloader import get_dataloader_and_loss_lower_bound
import torch
import numpy as np
import copy
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
import json
import yaml
import os
from torch.nn import functional as F
import wandb
import cProfile
import pstats
from torch.profiler import profile, record_function, ProfilerActivity

from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.training.dataloader import get_dataloader_and_loss_lower_bound
import torch
import numpy as np
import copy
from tqdm import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig
import json
import yaml
import os
from torch.nn import functional as F

import wandb

import cProfile
import pstats
from torch.profiler import profile, record_function, ProfilerActivity
from epsilon_transformers.training.logger import StructuredLogger



def train_epoch(model, optimizer, dataset, scheduler=None):
    model.train()

    epoch_losses = []

    for input_sequences, target_sequences in dataset:
        # Zero the gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        logits = model(input_sequences)

        # Reshape for loss calculation
        batch_size, seq_length, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_sequences.reshape(-1).to(torch.int64)

        # Compute loss
        loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = loss.reshape(batch_size, seq_length)

        # Backpropagation
        loss.mean().backward()

        # Perform optimization step
        optimizer.step()

        # Store the loss for this batch
        epoch_losses.append(loss.detach())

    if scheduler:
        # Pass the mean loss to the scheduler
        scheduler.step(torch.mean(torch.cat(epoch_losses)))

    # Compute and return the mean loss per context position across all batches
    return torch.concat(epoch_losses).mean(dim=0)

def validate_epoch(model, dataset):
    model.eval()

    with torch.no_grad():
        X, Y, probs = dataset.validation_data()
        logits = model(X)
        batch_size, seq_length, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = Y.reshape(-1).to(torch.int64)
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = loss.reshape(batch_size, seq_length)
        # multiply the loss (batch_size, seq_length) by the probabilities (batch_size) to get the weighted loss (batch_size, seq_length)
        loss = loss * probs.unsqueeze(1)
        return loss.sum(dim=0)

def save_model_config(logger, model):
    hooked_model_config_dict = copy.deepcopy(model.cfg.to_dict())
    hooked_model_config_dict['dtype'] = str(hooked_model_config_dict['dtype'])
    with open(os.path.join(logger.base_dir, 'hooked_model_config.json'), 'w') as f:
        json.dump(hooked_model_config_dict, f, indent=4)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(args):
    if args.device == 'cuda':
        if torch.cuda.is_available():
            return torch.device(f'cuda:{args.gpu_id}')
        else:
            print("CUDA is not available. Falling back to CPU.")
            return torch.device('cpu')
    elif args.device == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            print("MPS is not available. Falling back to CPU.")
            return torch.device('cpu')
    else:
        return torch.device('cpu')
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def main():
    parser = argparse.ArgumentParser(description='Train Transformer with specific hyperparameters.')
    parser.add_argument('--config', type=str, required=True, help='Path to run configuration file')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel CUDA execution')
    parser.add_argument('--gpu_id', type=int, default=0, required=False, help='GPU ID to use for this run')

    args = parser.parse_args()

    config = load_config(args.config)
    
    # Initialize the logger
    logger = StructuredLogger(config['experiment_dir'])
    
    # Wrap the entire main function with cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    if config['global_config']['wandb']:
        wandb.init(project=f"{config['global_config']['wandb_project']}_{config['global_config']['sweep_id']}",
                   name=config['run_id'])

    # Set device
    
    if args.parallel:
        device = f'cuda:{args.gpu_id}'
    else:
        device = config['global_config']['device']
    #print(f"Using device: {device}")

    # Parse process parameters

    dataloader, loss_lower_bound, d_vocab = get_dataloader_and_loss_lower_bound(
        process_params=config['process_config'],
        n_ctx=config['model_config']['n_ctx'],
        bos=config['train_config']['bos'],
        batches_per_epoch=config['train_config']['batches_per_epoch'],
        batch_size=config['train_config']['batch_size'],
        device=device,
    )

    np.savetxt('loss_lower_bound.txt', loss_lower_bound.cpu().numpy(), fmt='%f', delimiter=',', header='loss_lower_bound')

    config['model_config']['device'] = config['global_config']['device']
    config['model_config']['d_vocab'] = d_vocab
    config['model_config']['dtype'] = getattr(torch, config['model_config']['dtype'])

    hooked_model_config = HookedTransformerConfig(**config['model_config'])
    model = HookedTransformer(hooked_model_config)
    #model = torch.compile(model)
    logger.log({"status": "model loaded"})
    save_model_config(logger, model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train_config']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, cooldown=500, threshold=1e-6,
        verbose=True
    )
    print('MODEL DEVICE:', next(model.parameters()).device, type(next(model.parameters()).device))
    # print the device of the dataloader
    #print('DATALOADER DEVICE:', dataloader.device, type(dataloader.device))

    # PyTorch profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_stack=True, profile_memory=True) as prof:
        with record_function("model_training"):
            bar = tqdm(range(config['train_config']['n_epochs']), desc="Training", unit="epoch")

            # save initial model checkpoint
            
            num_tokens_seen = 0
            # do validation before starting the epoch loop
            val_loss_per_ctx_pos = validate_epoch(model, dataloader)
            val_loss_per_ctx_pos = val_loss_per_ctx_pos / loss_lower_bound
            mean_val_loss = val_loss_per_ctx_pos.mean().item()
            logger.log_epoch(-1, num_tokens_seen, 
                             None, 
                             val_loss_per_ctx_pos.tolist(), 
                             optimizer.param_groups[0]['lr'])
            logger.save_model_checkpoint(model, "0")

            for i in bar:
                loss_per_ctx_pos = train_epoch(model, optimizer, dataloader, scheduler) / loss_lower_bound
                mean_loss = loss_per_ctx_pos.mean().item()
                val_loss_per_ctx_pos = validate_epoch(model, dataloader) / loss_lower_bound
                mean_val_loss = val_loss_per_ctx_pos.mean().item()
                bar.set_postfix(loss=f"{mean_loss:.4f}", val_loss=f"{mean_val_loss:.4f}")

                num_tokens_seen += dataloader.tokens_per_epoch
                logger.save_model_checkpoint(model, f"{num_tokens_seen}")
                logger.log_epoch(i, num_tokens_seen, loss_per_ctx_pos.tolist(), val_loss_per_ctx_pos.tolist(), optimizer.param_groups[0]['lr'])

    # Print PyTorch profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(os.path.join(config['experiment_dir'], "pytorch_trace.json"))

    # Disable cProfile and print results
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)  # Print top 20 time-consuming functions
    stats.dump_stats(os.path.join(config['experiment_dir'], 'train_profile.prof'))

if __name__ == "__main__":
    main()