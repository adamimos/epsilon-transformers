import tempfile

import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from epsilon_transformers.process.dataset import (
    ProcessDataset,
    process_dataset_collate_fn,
)
from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.training.configs.training_configs import (
    LoggingConfig,
    OptimizerConfig,
    PersistanceConfig,
    ProcessDatasetConfig,
    TrainConfig,
)
from epsilon_transformers.training.train import (
    _check_if_action_batch,
    _set_random_seed,
    train_model,
)

# TODO: Paramaterize test_configs_throw_error_on_extra
# TODO: Patamaterize train_model w/ all configs
# TODO: Test mutually_exclusive_logs
# TODO: Test the log state (it get's reset when needed, it gets updated appropriately)
# TODO: Test for writing multiple checkpoints
# TODO: Write test to make sure transformer is initialized correctly


def test_configs_throw_error_on_extra():
    with pytest.raises(ValidationError):
        OptimizerConfig(
            optimizer_type="adam",
            learning_rate=1.06e-12,
            weight_decay=0.08,
            jungle_bunny="hoorah",
        )


def test_raw_model_config():
    model_config = RawModelConfig(
        d_vocab=2,
        d_model=100,
        n_ctx=45,
        d_head=48,
        n_head=12,
        d_mlp=12,
        n_layers=2,
    )
    model = model_config.to_hooked_transformer(seed=13, device="cpu")
    assert isinstance(model, HookedTransformer)
    input_tensor = torch.tensor([[0, 1, 0, 1, 1, 0]], device="cpu", dtype=torch.long)
    output = model(input_tensor)
    assert output.shape == torch.Size(
        [1, 6, 2]
    )  # batch, pos, vocab (it returns logits)


def test_dataloader_raw_hooked_transformer_compatibility():
    model_config = RawModelConfig(
        d_vocab=2,
        d_model=100,
        n_ctx=45,
        d_head=48,
        n_head=12,
        d_mlp=12,
        n_layers=2,
    )
    model = model_config.to_hooked_transformer(seed=13, device="cpu")

    dataset = ProcessDataset("ZeroOneR", {}, 10, 16)
    dataloader = DataLoader(
        dataset=dataset, collate_fn=process_dataset_collate_fn, batch_size=2
    )

    for x, _ in dataloader:
        output = model(x)
        assert output.shape == torch.Size(
            [1, 10, 2]
        )  # batch, pos, vocab (it returns logits)


def test_check_if_action_batch():
    assert (
        _check_if_action_batch(
            perform_action_every_n_tokens=100,
            batch_size=5,
            sequence_len=10,
            batch_idx=1,
        )
        == True
    )
    assert (
        _check_if_action_batch(
            perform_action_every_n_tokens=100,
            batch_size=5,
            sequence_len=10,
            batch_idx=2,
        )
        == False
    )
    with pytest.raises(AssertionError):
        _check_if_action_batch(
            perform_action_every_n_tokens=5, batch_size=2, sequence_len=10, batch_idx=4
        )


def test_train_and_test_dataloaders_are_different():
    _set_random_seed(45)

    config = ProcessDatasetConfig(process="z1r", batch_size=2, num_tokens=100)
    train_dataloader = config.to_dataloader(sequence_length=10)
    test_dataloader = config.to_dataloader(sequence_length=10)

    train_data = [x for x, _ in train_dataloader]
    test_data = [x for x, _ in test_dataloader]

    assert all([not torch.equal(x, y) for x, y in zip(train_data, test_data)])


def test_persistance():
    class LinearModel(torch.nn.Module):
        def __init__(self):
            super(LinearModel, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    model = LinearModel()
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PersistanceConfig(
            location="local",
            collection_location=temp_dir,
            checkpoint_every_n_tokens=100,
        )
        config.save_model(model, 55)

        loaded_model = LinearModel()
        loaded_model.fc.weight.data.fill_(1.0)
        loaded_model.fc.bias.data.fill_(0.0)
        assert any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(model.parameters(), loaded_model.parameters())
        )

        loaded_model.load_state_dict(torch.load(f"{temp_dir}/55.pt"))

    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(model.parameters(), loaded_model.parameters())
    )


def test_changing_log_states():
    # Input numbers make sense
    #
    raise NotImplementedError


def test_train_model_config_validator():
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_config = RawModelConfig(
                d_vocab=2,
                d_model=100,
                n_ctx=10,
                d_head=48,
                n_head=12,
                d_mlp=12,
                n_layers=2,
            )

            optimizer_config = OptimizerConfig(
                optimizer_type="adam", learning_rate=1.06e-4, weight_decay=0.8
            )

            dataset_config = ProcessDatasetConfig(
                process="mess3", batch_size=5, num_tokens=500, test_split=0.15
            )

            persistance_config = PersistanceConfig(
                location="local",
                collection_location=temp_dir,
                checkpoint_every_n_tokens=100,
            )

            TrainConfig(
                model=model_config,
                optimizer=optimizer_config,
                dataset=dataset_config,
                persistance=persistance_config,
                logging=LoggingConfig(
                    project_name="testing-logging-output", wandb=False
                ),
                verbose=True,
                seed=1337,
            )


def test_train_model():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_config = RawModelConfig(
            d_vocab=2,
            d_model=100,
            n_ctx=10,
            d_head=48,
            n_head=12,
            d_mlp=12,
            n_layers=2,
        )

        optimizer_config = OptimizerConfig(
            optimizer_type="adam", learning_rate=1.06e-4, weight_decay=0.8
        )

        dataset_config = ProcessDatasetConfig(
            process="rrxor", batch_size=5, num_tokens=500, test_split=0.15
        )

        persistance_config = PersistanceConfig(
            location="local",
            collection_location=temp_dir,
            checkpoint_every_n_tokens=100,
        )

        mock_config = TrainConfig(
            model=model_config,
            optimizer=optimizer_config,
            dataset=dataset_config,
            persistance=persistance_config,
            logging=LoggingConfig(project_name="testing-logging-output", wandb=False),
            verbose=True,
            seed=1337,
        )
        model, metrics = train_model(mock_config)

        # Assert that the loss has decreased


if __name__ == "__main__":
    test_train_model()
