from typing import Literal
from pydantic import BaseModel
import pathlib
import yaml
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig
from typing import Union

from epsilon_transformers.process.dataset import ProcessDataset, process_dataset_collate_fn

# TODO: Make Config ABS (??)

# TODO: Figure out if model seed should be it's own thing or whether we can just use the same seed across
# TODO: Decide on whether we want to use HookedTransformer exclusively or whether creating our own model class makes the most sense

# TODO: Add a learning rate scheduler config
# TODO: Add a WandbLoggingConfig
# TODO: Add a sweep config
# TODO: Add epoch training

class Config(BaseModel, extra='forbid'):
    @classmethod
    def from_yaml(cls, config_path: pathlib.Path) -> 'Config':
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)

class RawModelConfig(Config):
    d_vocab: int
    d_model: int
    n_ctx: int
    d_head: int
    n_head: int
    d_mlp: int
    n_layers: int

    def to_hooked_transformer(self, seed: int, device: torch.device) -> HookedTransformer:
        config = HookedTransformerConfig(
        d_model=self.d_model,
        d_head=self.d_head,
        n_layers=self.n_layers,
        n_ctx=self.n_ctx,
        n_heads=self.n_head,
        d_mlp=self.d_mlp,
        d_vocab=self.d_vocab,
        seed=seed,
        device=device,
        act_fn='relu'
        )
        return HookedTransformer(config)

Optimizer = Union[torch.optim.Adam, torch.optim.SGD]

class OptimizerConfig(Config):
    optimizer_type: Literal['sgd', 'adam']
    learning_rate: float
    weight_decay: float

    def from_model(self, model: torch.nn.Module, device: torch.device) -> Optimizer:
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD
        else:
            raise ValueError(f"{self.optimizer_type} is not a valid optimizer_type. It must be either 'adam' or 'sgd'")

        return optimizer(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

class PersistanceConfig(Config):
    location: Literal['local', 'gdrive']
    path: pathlib.Path
    checkpoint_every_n_tokens: int

    def save_model(self, model: torch.nn.Module):
        raise NotImplementedError

class ProcessDatasetConfig(Config):
    process: str
    batch_size: int
    num_tokens: int
    
    def to_dataloader(self, sequence_length: int) -> DataLoader:
        dataset = ProcessDataset(process_name=self.process, sequence_length=sequence_length, num_samples=self.num_tokens)
        return DataLoader(dataset=dataset, collate_fn=process_dataset_collate_fn, batch_size=self.batch_size)

class TrainConfig(Config):
    model: RawModelConfig
    optimizer: OptimizerConfig
    dataset: ProcessDatasetConfig
    persistance: PersistanceConfig
    seed: int
