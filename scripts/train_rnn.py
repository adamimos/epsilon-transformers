import argparse
from epsilon_transformers.training.logger import StructuredLogger
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.training.dataloader import get_dataloader_and_loss_lower_bound_from_process, get_dataloader_from_data
import torch
import numpy as np
import copy
from tqdm import tqdm

from torch import nn

import json
import yaml
import os
from torch.nn import functional as F
from epsilon_transformers.training.generate_data import load_process_data

import wandb

class RNNWrapper(nn.Module):
    def __init__(self, rnn, output_layer):
        super().__init__()
        self.rnn = rnn
        self.output_layer = output_layer
    
    def forward(self, x):
        output, hidden = self.rnn(x)
        return self.output_layer(output)
    
    def forward_with_hidden(self, x, hidden=None):
        """Run forward pass and return both output and hidden states"""
        output, hidden = self.rnn(x, hidden)
        return self.output_layer(output), hidden

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

    #if scheduler:
        # Pass the mean loss to the scheduler
        #scheduler.step(torch.mean(torch.cat(epoch_losses)))

    # Compute and return the mean loss per context position across all batches
    return torch.concat(epoch_losses).mean(dim=0)

def train_epoch_all(model, optimizer, dataset, scheduler=None):
    model.train()

    X, Y, probs = dataset.validation_data()
    logits = model(X)
    batch_size, seq_length, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = Y.reshape(-1).to(torch.int64)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    loss = loss.reshape(batch_size, seq_length)
    loss = loss * probs.unsqueeze(1)
    loss = loss.sum(dim=0)
    loss.mean().backward()
    optimizer.step()
    #if scheduler:
    #    scheduler.step(loss.mean())
    return loss

def validate_epoch_all(model, dataset, scheduler=None):
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
        if scheduler:
            scheduler.step(loss.mean())
            #scheduler.step()
        return loss.sum(dim=0)

def validate_epoch_sample(model, dataset):
    pass
    # TODO: implement validate_epoch_sample

def save_model_config(logger, config):
    """Save the model configuration to a JSON file"""
    model_config = copy.deepcopy(config['model_config'])
    # Convert any non-serializable types to strings
    model_config['dtype'] = str(model_config['dtype'])
    
    with open(os.path.join(logger.base_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)

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
    
def create_model(config, vocab_size, device):
    rnn_type = config['model_config'].get('rnn_type', 'LSTM').upper()
    rnn_class = getattr(nn, rnn_type)
    
    rnn = rnn_class(
        input_size=vocab_size,
        hidden_size=config['model_config']['hidden_size'],
        num_layers=config['model_config']['num_layers'],
        dropout=config['model_config'].get('dropout', 0.0),
        batch_first=True,
        bidirectional=config['model_config'].get('bidirectional', False)
    ).to(device)
    
    output_size = config['model_config']['hidden_size'] * 2 if config['model_config'].get('bidirectional', False) else config['model_config']['hidden_size']
    output_layer = nn.Linear(output_size, vocab_size).to(device)
    
    return RNNWrapper(rnn, output_layer)

def main():
    parser = argparse.ArgumentParser(description='Train Transformer with specific hyperparameters.')
    parser.add_argument('--config', type=str, required=True, help='Path to run configuration file')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel CUDA execution')
    parser.add_argument('--gpu_id', type=int, default=0, required=False, help='GPU ID to use for this run')

    args = parser.parse_args()

    config = load_config(args.config)
    logger = StructuredLogger(config['experiment_dir'])
    set_seed(42)

    if config['global_config']['wandb']:
        wandb.init(project=f"{config['global_config']['wandb_project']}_{config['global_config']['sweep_id']}",
                   name=config['run_id'])

    # Set device
    
    if args.parallel:
        device = f'cuda:{args.gpu_id}'
    else:
        device = config['global_config']['device']
    #print(f"Using device: {device}")

    val_every = config['global_config']['val_every']
    train_type = config['train_config'].get('train_type', 'normal')
    save_every = config['global_config'].get('save_every', 1)
    # Parse process parameters

    # Try to load pre-generated data
    process_data = load_process_data(config, config['global_config']['process_dir'])
    print("process_data:", process_data)
    
    if process_data is not None:
        # Data was pre-generated, load it
        dataloader, d_vocab = get_dataloader_from_data(
            process_data['transformer_inputs'],
            process_data['probs'],
            config['train_config']['batches_per_epoch'],
            config['train_config']['batch_size'],
            device
        )
        loss_lower_bound = torch.from_numpy(process_data['loss_lower_bound']).to(device)
    else:
        # Data wasn't pre-generated, generate it now
        dataloader, loss_lower_bound, d_vocab = get_dataloader_and_loss_lower_bound_from_process(
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

    model = create_model(config, d_vocab, device)
    #model = torch.compile(model)
    logger.log({"status": "model loaded"})
    save_model_config(logger, config)
    if train_type == 'all':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['train_config']['learning_rate'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train_config']['learning_rate'])
    
    if config['global_config']['scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1000, cooldown=200, threshold=1e-6,
            verbose=True
        )
        # implement cosine annealing warm restarts
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #    optimizer, T_0=200, T_mult=2, eta_min=1e-7, last_epoch=-1, verbose=True
        #)
    else:
        scheduler = None
    print('MODEL DEVICE:', next(model.parameters()).device, type(next(model.parameters()).device))
    # print the device of the dataloader
    #print('DATALOADER DEVICE:', dataloader.device, type(dataloader.device))
    bar = tqdm(range(config['train_config']['n_epochs']), desc="Training", unit="epoch")

    # save initial model checkpoint
    
    num_tokens_seen = 0
    # do validation before starting the epoch loop

    val_loss_per_ctx_pos = validate_epoch_all(model, dataloader)
    val_loss_per_ctx_pos = val_loss_per_ctx_pos / loss_lower_bound
    mean_val_loss = val_loss_per_ctx_pos.mean().item()
    logger.log_epoch(-1, num_tokens_seen, 
                     None, 
                     val_loss_per_ctx_pos.tolist(), 
                     optimizer.param_groups[0]['lr'])
    logger.save_model_checkpoint(model, "0")

    for i in bar:
        if train_type == 'all':
            loss_per_ctx_pos = train_epoch_all(model, optimizer, dataloader, scheduler) / loss_lower_bound
        else:
            loss_per_ctx_pos = train_epoch(model, optimizer, dataloader, scheduler) / loss_lower_bound
        mean_loss = loss_per_ctx_pos.mean().item()
        
        if val_every is not None and i % val_every == 0:
            # TODO: implement logic for val_type
            val_loss_per_ctx_pos = validate_epoch_all(model, dataloader, scheduler) / loss_lower_bound
            mean_val_loss = val_loss_per_ctx_pos.mean().item()
            bar.set_postfix(loss=f"{mean_loss:.4f}", val_loss=f"{mean_val_loss:.4f}")
        else:
            val_loss_per_ctx_pos = None
            bar.set_postfix(loss=f"{mean_loss:.4f}")

        num_tokens_seen += dataloader.tokens_per_epoch
        if save_every is not None and i % save_every == 0:
            logger.save_model_checkpoint(model, f"{num_tokens_seen}")
        logger.log_epoch(i, num_tokens_seen, loss_per_ctx_pos.tolist(), 
                         val_loss_per_ctx_pos.tolist() if val_loss_per_ctx_pos is not None else None, 
                         optimizer.param_groups[0]['lr'])

if __name__ == "__main__":
    main()