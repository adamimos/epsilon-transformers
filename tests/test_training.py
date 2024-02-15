import pytest
from pydantic import ValidationError

from epsilon_transformers.training.configs import TrainConfig, RawModelConfig, OptimizerConfig, ProcessDatasetConfig, PersistanceConfig
from epsilon_transformers.training.train import train_model

# TODO: Paramaterize test_configs_throw_error_on_extra

def test_configs_throw_error_on_extra():
    with pytest.raises(ValidationError):
        OptimizerConfig(optimizer_type='adam', learning_rate=1.06e-12, weight_decay=.08, jungle_bunny='hoorah')
    
if __name__ == "__main__":
    test_configs_throw_error_on_extra()