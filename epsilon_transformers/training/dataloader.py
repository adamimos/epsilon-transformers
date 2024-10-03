from typing import Tuple
import torch
from epsilon_transformers.process.Process import GHMM
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.process.processes import TransitionMatrixGHMM
from torch.utils.data import IterableDataset

def generate_all_seqs(process: GHMM, seq_len: int, bos: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate all possible sequences and their probabilities for a given process and sequence length.

    Args:
        process (Process): The process to generate sequences from.
        seq_len (int): The length of sequences to generate.
        bos (bool): Whether to include a beginning-of-sequence token. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - transformer_inputs: All possible sequences.
            - probs: Probability of each sequence.
            - loss_lower_bound: Loss lower bound for each context window position.

    Raises:
        ValueError: If the sum of probabilities is not approximately 1.0.
    """
    # Adjust depths and lengths based on BOS token
    msp_depth = seq_len + (1 if bos else 2)
    final_seq_len = seq_len - (1 if bos else 0)
    bos_token = process.vocab_len if bos else None

    # Generate Mixed State Presentation
    msp = process.derive_mixed_state_tree(depth=msp_depth)
    paths, probs = msp.get_paths_and_probs(depth=final_seq_len)
    myopic_entropy = msp.myopic_entropy

    # Convert to tensors
    transformer_inputs = torch.tensor(paths, dtype=torch.int32)
    probs = torch.tensor(probs, dtype=torch.float32)

    # Add BOS token if required
    if bos:
        bos_column = torch.full((len(paths), 1), bos_token, dtype=torch.int32)
        transformer_inputs = torch.cat([bos_column, transformer_inputs], dim=1)
        loss_lower_bound = myopic_entropy[:-1]
    else:
        loss_lower_bound = myopic_entropy[1:-1]

    # Validate probabilities
    if not torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6):
        raise ValueError(f"Sum of probabilities is {probs.sum().item():.6f}, expected 1.0")

    return transformer_inputs, probs, loss_lower_bound
class BatchGenerator(IterableDataset):
    def __init__(self, transformer_inputs, probs, batches_per_epoch, batch_size, device):
        self.transformer_inputs = transformer_inputs.to(device)
        self.probs = probs.to(device)
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            # Sample indices based on probabilities
            sample_inds = torch.multinomial(self.probs, self.batch_size, replacement=True)
            batch = self.transformer_inputs[sample_inds]
            X, Y = batch[:, :-1], batch[:, 1:]
            yield X, Y
"""
class BatchGenerator:
    def __init__(self, transformer_inputs, probs, batches_per_epoch, batch_size):
        self.transformer_inputs = transformer_inputs
        self.probs = probs
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        total_samples = self.batches_per_epoch * self.batch_size
        sample_inds = torch.multinomial(self.probs, total_samples, replacement=True)
        sample_inds = sample_inds.reshape(self.batches_per_epoch, self.batch_size)

        for batch_indices in sample_inds:
            batch = self.transformer_inputs[batch_indices]
            X, Y = batch[:, :-1], batch[:, 1:]
            yield X, Y

    def validation_data(self):
        total_samples = self.transformer_inputs.shape[0]
        for start_idx in range(0, total_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_samples)
            batch = self.transformer_inputs[start_idx:end_idx]
            batch_probs = self.probs[start_idx:end_idx]
            X, Y = batch[:, :-1], batch[:, 1:]
            yield X, Y, batch_probs
"""

def get_dataloader_and_loss_lower_bound(process_params: dict, n_ctx: int, bos: bool, batches_per_epoch: int, batch_size: int, device: str) -> Tuple[BatchGenerator, torch.Tensor]:
    # Initialize the process
    T = get_matrix_from_args(**process_params)
    ghmm = TransitionMatrixGHMM(T)
    ghmm.name = process_params['name']
    print("Process initialized successfully!")

    all_seqs, all_seqs_probs, loss_lower_bound = generate_all_seqs(ghmm, n_ctx+1, bos)
    all_seqs = all_seqs.to(device)
    all_seqs_probs = all_seqs_probs.to(device)
    loss_lower_bound = torch.tensor(loss_lower_bound, dtype=torch.float32).to(device)

    dataloader = BatchGenerator(all_seqs, all_seqs_probs, batches_per_epoch, batch_size, device)

    d_vocab = T.shape[0] + (1 if bos else 0)
    d_vocab = d_vocab 

    return dataloader, loss_lower_bound, d_vocab