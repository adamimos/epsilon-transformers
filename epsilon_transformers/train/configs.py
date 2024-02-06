from typing import Literal
from pydantic import BaseModel
from torch.optim import Adam, SGD
import pathlib

from epsilon_transformers.processes.process import Process
from epsilon_transformers.nn.simple_transformer import MultilayerTransformer

# TODO: Make a process registry
# TODO: Add a learning rate scheduler config
# TODO: Make from_yaml be inherited from a base config class

class RawModelConfig(BaseModel):
    d_vocab: int
    d_model: int
    input_size: int
    d_head: int
    n_head: int
    d_mlp: int
    n_layers: int
    use_layernorm: bool
    
    def from_yaml() -> MultilayerTransformer:
        raise NotImplementedError

class OptimizerConfig(BaseModel):
    optimizer_type: Literal['sgd', 'adam']
    learning_rate: float
    weight_decay: float

    def from_yaml() -> Adam | SGD:
        raise NotImplementedError

from typing import Any
class WandbConfig(BaseModel):
    foo: Any

class PersistanceConfig(BaseModel):
    bar: Literal['local', 'gdrive']
    path: pathlib.Path

import yaml
class TrainConfig(BaseModel):
    model_config: RawModelConfig
    optimizer_config: OptimizerConfig
    wandb_config: WandbConfig
    process: str
    batch_size: int
    num_tokens: int
    checkpoint_every_n_tokens: int
    persistance_config: PersistanceConfig
    epochs: int
    seed: int

    def from_yaml(config_path: pathlib.Path) -> 'TrainConfig':
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return TrainConfig(**config_data)

from typing import Any
class SweepConfig(BaseModel):
    foo: Any
