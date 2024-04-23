from io import BytesIO
from typing import Literal
from pydantic import BaseModel, model_validator
import pathlib
import yaml
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig
from typing import Union, Optional
import wandb
import os
import dotenv
import math
from dataclasses import dataclass, asdict

from epsilon_transformers.persistence import LocalPersister, Persister, S3Persister
from epsilon_transformers.process.processes import PROCESS_REGISTRY
from epsilon_transformers.process.dataset import (
    ProcessDataset,
    process_dataset_collate_fn,
)
from epsilon_transformers.training.configs.base_config import Config
from epsilon_transformers.training.configs.model_configs import RawModelConfig


# TODO: For persistence config, upon init make sure that you check that the relevant environment variables are set
# TODO: Generalize the checkpoint_dir option so that it can work w/ S3 outputs

# TODO: Make Config ABS (??)
# TODO: Turn log input into a dataclass (??)
# TODO: Have a no persistenc config option

# TODO: Put all the functionality of the log congig into the logger
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


Optimizer = Union[torch.optim.Adam, torch.optim.SGD]


class OptimizerConfig(Config):
    optimizer_type: Literal["sgd", "adam"]
    learning_rate: float
    weight_decay: float

    def from_model(self, model: torch.nn.Module, device: torch.device) -> Optimizer:
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise ValueError(
                f"{self.optimizer_type} is not a valid optimizer_type. It must be either 'adam' or 'sgd'"
            )

        return optimizer(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )


class PersistanceConfig(Config):
    location: Literal["local", "s3"]
    collection_location: pathlib.Path | str
    checkpoint_every_n_tokens: int

    def init(self) -> Persister:
        if self.location == "local":
            return LocalPersister(collection_location=self.collection_location)
        elif self.location == "s3":
            return S3Persister(collection_location=self.collection_location)
        else:
            raise ValueError(
                f"{self.location} is an invalid location value. It must be either 'local' or 's3'"
            )


class ProcessDatasetConfig(Config):
    process: str
    batch_size: int
    num_tokens: int
    test_split: float

    def to_dataloader(self, sequence_length: int, train: bool) -> DataLoader:
        dataset = ProcessDataset(
            process_name=self.process,
            sequence_length=sequence_length,
            num_samples=(
                self.num_tokens
                if train
                else math.floor(self.num_tokens * self.test_split)
            ),
        )
        return DataLoader(
            dataset=dataset,
            collate_fn=process_dataset_collate_fn,
            batch_size=self.batch_size,
        )


@dataclass
class Log:
    train_loss: Optional[float]
    test_loss: Optional[float]
    config: "LoggingConfig"

    def reset(self):
        if self.config.train_loss:
            self.train_loss = 0.0
        else:
            self.train_loss = None

        if self.config.test_loss:
            self.test_loss = 0.0
        else:
            self.test_loss = None

    def update_metrics(self, train_or_test: Literal["train", "test"], loss: float):
        if train_or_test == "train" and self.config.test_loss:
            self.train_loss += loss
        elif train_or_test == "test" and self.config.train_loss:
            self.test_loss += loss
        else:
            raise ValueError

    def persist(self):
        if self.config.wandb:
            wandb.log({k: v for k, v in asdict(self).items() if v is not None and not isinstance(v, LoggingConfig)})
        if self.config.local is not None:
            raise NotImplementedError

class LoggingConfig(Config):
    local: Optional[pathlib.Path] = None
    wandb: bool = True
    project_name: Optional[str]
    train_loss: bool = True
    test_loss: bool = True

    def close(self):
        if self.wandb:
            wandb.finish()
        if self.local is not None:
            raise NotImplementedError

    def init(self) -> Log:
        return Log(
            config=self,
            train_loss=0.0 if self.train_loss else None,
            test_loss=0.0 if self.test_loss else None,
        )


class TrainConfig(Config):
    model: RawModelConfig
    optimizer: OptimizerConfig
    dataset: ProcessDatasetConfig
    persistance: PersistanceConfig
    logging: LoggingConfig
    seed: int
    verbose: bool

    @model_validator(mode='after')
    def validate_model(self):
        dataset_process = self.dataset.process
        if dataset_process:
            process_vocab_len = PROCESS_REGISTRY[dataset_process].vocab_len
            if self.model.d_vocab != process_vocab_len:
                raise ValueError(f"Model's d_vocab ({self.model.d_vocab}) doesn't match dataset process's vocab_len ({process_vocab_len})")
        return self

    def init_logger(self) -> Log:
        if self.logging.wandb:
            dotenv.load_dotenv()
            wandb_api_key = os.environ.get("WANDB_API_KEY", None)
            if wandb_api_key is None:
                raise ValueError(
                    "To use wandb, set your API key as the environment variable `WANDB_API_KEY`"
                )

            wandb.login(key=wandb_api_key)
            wandb.init(project=self.logging.project_name, config=self.model_dump())
        if self.logging.local is not None:
            raise NotImplementedError()
        return self.logging.init()
