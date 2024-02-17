from typing import Literal
from pydantic import BaseModel
import pathlib
import yaml
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig
from typing import Union, Optional, Dict
import wandb
import os
import dotenv
from dataclasses import dataclass

from epsilon_transformers.process.dataset import ProcessDataset, process_dataset_collate_fn

# TODO: Make Config ABS (??)
# TODO: Turn log input into a dataclass (??)

# TODO: Fix the eval_dataloader_ratio_creation
# TODO: Create a logger & log the file path and intermediary metrics
# TODO: Add validator to make sure test_split is a fraction
# TODO: Add validator in Persistence Config to make sure the path is a dir
# TODO: Add validator in Logging Config to make sure that if we're logging wandb then we're using a project name
# TODO: Figure out if model seed should be it's own thing or whether we can just use the same seed across
# TODO: Decide on whether we want to use HookedTransformer exclusively or whether creating our own model class makes the most sense

# TODO: Think if you can make Log DRY
# TODO: Switch statement code smell with update_loss_metrics

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
    checkpoint_dir: pathlib.Path
    checkpoint_every_n_tokens: int

    def save_model(self, model: torch.nn.Module, num_tokens: int):
        if self.location == 'local':
            save_path = self.checkpoint_dir / f"{num_tokens}.pt"
            torch.save(model.state_dict(), save_path)
        elif self.location == 'gdrive':
            raise NotImplementedError
        else:
            raise ValueError(f"{self.location} is an invalid location value. It must be either 'local' or 'gdrive'")

class ProcessDatasetConfig(Config):
    process: str
    batch_size: int
    num_tokens: int
    test_split: float
    
    def to_dataloader(self, sequence_length: int, train: bool) -> DataLoader:
        dataset = ProcessDataset(
            process_name=self.process, 
            sequence_length=sequence_length, 
            num_samples=self.num_tokens if train else self.num_tokens * self.test_split
        )
        return DataLoader(dataset=dataset, collate_fn=process_dataset_collate_fn, batch_size=self.batch_size)

@dataclass
class Log:
    train_loss: Optional[float]
    train_accuracy: Optional[float]
    test_loss: Optional[float]
    test_accuracy:Optional[float]

    def merge_mutually_exclusive_logs(self, log: 'Log') -> 'Log':
        merged_data = dict()
        for field in self.__annotations__:
            value_1 = getattr(self, field)
            value_2 = getattr(log, field)

            if value_1 is not None and value_2 is not None:
                raise ValueError(f"Logs are not mutually exclusive. Conflict in field '{field}': {value_1} and {value_2}")

            if value_1 is not None:
                merged_data[field] = value_1
            else:
                merged_data[field] = value_2
        return Log(**merged_data)

class LoggingConfig(Config):
    local: Optional[pathlib.Path] = None
    wandb: bool = True

    project_name: Optional[str]

    train_loss: bool = True
    train_accuracy: bool = False
    test_loss: bool = True
    test_accuracy:bool = False

    def log(self, info: Dict[str, float]):
        if self.wandb:
            # TODO: Make sure that the dict has all the expected entries
            wandb.log(info)
        if self.local is not None:
            raise NotImplementedError
    
    def close(self):
        if self.wandb:
            wandb.finish()
        if self.local is not None:
            raise NotImplementedError
    
    def init(self) -> Log:
        return Log(
            train_loss=0.0 if self.train_loss else None,
            train_accuracy=0.0 if self.train_accuracy else None,
            test_loss=0.0 if self.test_loss else None,
            test_accuracy=0.0 if self.test_accuracy else None
        )
    
    def update_train_metrics(self, log: Log, loss: float) -> Log:
        if self.train_loss:
            log.train_loss += loss
        if self.train_accuracy:
            raise NotImplementedError
        return log

    def update_test_metrics(self, log: Log, loss: float) -> Log:
        if self.test_loss:
            log.test_loss += loss
        if self.test_accuracy:
            raise NotImplementedError
        return log

class TrainConfig(Config):
    model: RawModelConfig
    optimizer: OptimizerConfig
    dataset: ProcessDatasetConfig
    persistance: PersistanceConfig
    logging: LoggingConfig
    seed: int

    def init_logger(self) -> Log:
        if self.logging.wandb:
            dotenv.load_dotenv()
            wandb_api_key = os.environ.get("WANDB_API_KEY", None)
            if wandb_api_key is None:
                raise ValueError("To use wandb, set your API key as the environment variable `WANDB_API_KEY`")

            wandb.login(key=wandb_api_key)
            wandb.init(
                project=self.logging.project_name,
                config=self.model_dump()
            )
        if self.logging.local is not None:
            raise NotImplementedError()
        return self.logging.init()

    