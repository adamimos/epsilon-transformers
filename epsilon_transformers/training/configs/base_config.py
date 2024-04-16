import pathlib
from pydantic import BaseModel
import yaml


class Config(BaseModel, extra="forbid"):
    @classmethod
    def from_yaml(cls, config_path: pathlib.Path) -> "Config":
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)

