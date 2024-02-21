from typing import List, Set, Tuple
import pytest
from torch.utils.data import DataLoader
import numpy as np

from epsilon_transformers.process.MixedStatePresentation import MixedStateTreeNode
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

def test_compute_emission_probabilities():
    return NotImplementedError

def test_compute_next_distribution():
    return NotImplementedError

def test_msp_creation():
    process = ZeroOneR()
    z1r_msp = process.derive_mixed_state_presentation(max_depth=5)
    paths_and_probs = sorted([(x.path, x.state_prob_vector) for x in z1r_msp.nodes], key=lambda x: len(x[0]))
    assert paths_and_probs[0][0] == [] and np.array_equal(paths_and_probs[0][1], np.array([1/3, 1/3, 1/3]))
    def _query_path(list_of_tuples, path: list) -> Tuple[List[int], np.ndarray]:
        for tup in list_of_tuples:
            if path == tup[0]:
                return tup
        raise ValueError(f"{path} not in list of tuples")
    assert np.array_equal(_query_path(paths_and_probs, [])[1], np.array([1/3, 1/3, 1/3]))
    assert np.array_equal(_query_path(paths_and_probs, [0])[1], np.array([1/3, 2/3, 0]))
    assert np.array_equal(_query_path(paths_and_probs, [1])[1], np.array([1/3, 0, 2/3]))
    assert np.array_equal(_query_path(paths_and_probs, [1, 0])[1], np.array([1/2, 1/2, 0]))
    assert all([np.all((vector == 0) | (vector == 1)) and np.sum(vector) == 1 for path, vector in paths_and_probs if path != [] and path != [0] and path != [1] and path != [1,0]])

if __name__ == "__main__":
    test_msp_creation()
