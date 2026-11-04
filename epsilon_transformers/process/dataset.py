import numpy as np
import torch
from typing import Iterator
from jaxtyping import Float
from torch.utils.data import IterableDataset

from epsilon_transformers.process.Process import Process
from epsilon_transformers.process.processes import PROCESS_REGISTRY

# TODO: Create a custom dataloader so you don't have to import the collate_function everywehre
# TODO: Assert they are in the correct vocabulary
# TODO: Make the dataset parallel distributed (??)
# TODO: Figure out the device allocation for batching


class ProcessDataset(IterableDataset):
    """Windows of sequence_length + 1 emitted tokens, yielded as (input, target) pairs shifted by one token.

    Each window starts in a hidden state drawn independently from the process's stationary distribution, so
    every window is a stationary sample of the process. (Cutting consecutive windows from one long emission
    stream gives stationary windows only when the window length is not commensurate with a periodicity of
    the process. For example, RRXOR emits two random bits and their XOR in a fixed cycle of three, so with
    sequence_length = 20 every 21-token window would start at the same phase of the cycle and the training
    data would be phase-locked.)
    """

    process: Process
    stationary_distribution: np.ndarray
    sequence_length: int
    num_samples: int

    def __init__(
        self,
        process_name: str,
        process_params: dict[str, float],
        sequence_length: int,
        num_samples: int,
    ):
        super().__init__()

        process_class = PROCESS_REGISTRY.get(process_name, None)
        if process_class is None:
            raise ValueError(
                f"{process_name} is not a recognized process. It must be one of the following {PROCESS_REGISTRY.keys()}"
            )
        self.process = process_class(**process_params)
        stationary = np.clip(self.process.steady_state_vector, 0.0, None)
        self.stationary_distribution = stationary / stationary.sum()
        self.sequence_length = sequence_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self) -> Iterator[tuple[list[int], list[int]]]:
        for _ in range(self.num_samples):
            start_state = np.random.choice(
                self.process.num_states, p=self.stationary_distribution
            )
            process_history = list(
                self.process.yield_emissions(
                    sequence_len=self.sequence_length + 1,
                    current_state_idx=start_state,
                )
            )
            yield (process_history[:-1], process_history[1:])


def process_dataset_collate_fn(
    batch: list[tuple[list[int], list[int]]],
) -> tuple[
    Float[torch.Tensor, "batch_size sequence_length"],
    Float[torch.Tensor, "batch_size sequence_length"],
]:
    data = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
