import pytest
from pydantic import ValidationError
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer


from epsilon_transformers.process.dataset import ProcessDataset, process_dataset_collate_fn
from epsilon_transformers.training.configs import TrainConfig, RawModelConfig, OptimizerConfig, ProcessDatasetConfig, PersistanceConfig
from epsilon_transformers.training.train import train_model, _check_if_action_batch, _set_random_seed

# TODO: Double check the HookedTransformer implementation and make sure that model(tokens) actually returns, and that there isn't something deeply wrong 
# TODO: Paramaterize test_configs_throw_error_on_extra

def test_configs_throw_error_on_extra():
    with pytest.raises(ValidationError):
        OptimizerConfig(optimizer_type='adam', learning_rate=1.06e-12, weight_decay=.08, jungle_bunny='hoorah')

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
    model = model_config.to_hooked_transformer(seed=13, device='cpu')
    assert isinstance(model, HookedTransformer)
    input_tensor = torch.tensor([[0,1,0,1,1,0]], device='cpu', dtype=torch.long)
    output = model(input_tensor)
    assert output.shape == torch.Size([1,6,2]) # batch, pos, vocab (it returns logits)

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
    model = model_config.to_hooked_transformer(seed=13, device='cpu')
    
    dataset = ProcessDataset('z1r', 10, 16)
    dataloader = DataLoader(dataset=dataset, collate_fn=process_dataset_collate_fn, batch_size=2)

    for x, _  in dataloader:
        output = model(x)
        assert output.shape == torch.Size([1,10,2]) # batch, pos, vocab (it returns logits)

def test_check_if_action_batch():
    # Test case 1: Valid scenario where action should be performed on the last batch
    assert _check_if_action_batch(perform_action_every_n_tokens=100, batch_size=5, sequence_len=10, batch_idx=1) == True
    # Test case 2: Valid scenario where action should not be performed
    assert _check_if_action_batch(perform_action_every_n_tokens=100, batch_size=5, sequence_len=10, batch_idx=2) == False
    # Test case 4: Invalid scenario where perform_action_every_n_tokens is not greater than tokens_per_batch
    with pytest.raises(AssertionError):
        _check_if_action_batch(perform_action_every_n_tokens=5, batch_size=2, sequence_len=10, batch_idx=4)

def test_train_and_test_dataloaders_are_different():
    _set_random_seed(45)

    config = ProcessDatasetConfig(process='z1r', batch_size=2, num_tokens=100)
    train_dataloader = config.to_dataloader(sequence_length=10)
    test_dataloader = config.to_dataloader(sequence_length=10)
    
    train_data = [x for x, _ in train_dataloader]
    test_data = [x for x, _ in test_dataloader]

    assert all([not torch.equal(x, y) for x,y in zip(train_data, test_data)])

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
        checkpoint_dir="some/random/path",
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
    test_train_and_test_dataloaders_are_different()