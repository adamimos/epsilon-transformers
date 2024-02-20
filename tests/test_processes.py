import pytest
from torch.utils.data import DataLoader

from epsilon_transformers.process.Process import ProcessHistory
from epsilon_transformers.process.processes import ZeroOneR
from epsilon_transformers.process.dataset import ProcessDataset, process_dataset_collate_fn

# TODO: Check for off by 1 error in the sample_emission asserts
# TODO: Check that histogram distribution matches steady state distribution in test_generate_single_sequence

# TODO: parameterize this for each process
# TODO: parameterize current_state_index
# TODO: Parameterize number of sequences & sequence lengths

def test_sample_emission():
    process = ZeroOneR()

    # Test when an invalid current state index is provided
    current_state_index = -1  # Replace with an invalid index
    with pytest.raises(AssertionError):
        process._sample_emission(current_state_index)

    # Test when a valid current state index is provided
    current_state_index = 2  # Replace with a valid index
    emission = process._sample_emission(current_state_index)
    assert 0 <= emission <= process.vocab_len

def test_generate_process_history():
    process = ZeroOneR()

    outs = process.generate_process_history(12)
    assert len(outs) == 12
    assert isinstance(outs, ProcessHistory)

def test_process_dataset():
    dataset = ProcessDataset('z1r', 10, 15)
    
    for data, label in dataset:
        assert len(data) == len(label) == 10
        assert data[1:] == label[:-1]

    dataset = ProcessDataset('z1r', 10, 16)
    dataloader = DataLoader(dataset=dataset, collate_fn=process_dataset_collate_fn, batch_size=2)

    for data, label in dataloader:
        assert len(data) == len(label) == 2  # Since batch_size is set to 2
        assert (data[:, 1:] == label[:, :-1]).all()

# TODO: Mock z1r
def test_msp_creation():
    raise NotImplementedError