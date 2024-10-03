from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml
import uuid
from datetime import datetime
@dataclass
class ProcessConfig:
    name: str
    params: Dict[str, Any]

@dataclass
class ModelConfig:
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    n_ctx: int

@dataclass
class SweepConfig:
    learning_rates: List[float]
    batch_sizes: List[int]
    processes: List[ProcessConfig]

@dataclass
class GlobalConfig:
    output_dir: str
    num_gpus: int

@dataclass
class ExperimentConfig:
    global_config: GlobalConfig
    model_config: ModelConfig
    sweep_config: SweepConfig
    run_id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    experiment_name: str = field(default_factory=lambda: f"model_{ModelConfig.n_layers}L_{ModelConfig.d_model}D_{ModelConfig.n_heads}H_{ProcessConfig.name}")
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ExperimentConfig(**config_dict)
