import pytest
from pydantic import ValidationError

from epsilon_transformers.training.configs import TrainConfig, RawModelConfig, OptimizerConfig, ProcessDatasetConfig, PersistanceConfig
from epsilon_transformers.training.train import train_model

# TODO: Paramaterize test_configs_throw_error_on_extra

def test_configs_throw_error_on_extra():
    with pytest.raises(ValidationError):
        OptimizerConfig(optimizer_type='adam', learning_rate=1.06e-12, weight_decay=.08, jungle_bunny='hoorah')

def test_train_model():
    model_config = RawModelConfig(
        d_vocab=2,
        d_model=100,
        n_ctx=45,
        d_head=48,
        n_head=12,
        d_mlp=12,
        n_layers=2,
    )

    optimizer_config = OptimizerConfig(
        optimizer_type='adam',
        learning_rate=1.06e-4,
        weight_decay=0.8
    )

    dataset_config = ProcessDatasetConfig(
        process='z1r',
        batch_size=2,
        num_tokens=1000
    )

    persistance_config = PersistanceConfig(
        location='local',
        path="some/random/path",
        checkpoint_every_n_tokens=100
    )

    mock_config = TrainConfig(
        model=model_config,
        optimizer=optimizer_config,
        dataset=dataset_config,
        persistance=persistance_config,
        seed=1337
    )
    train_model(mock_config)

if __name__ == "__main__":
    test_train_model()