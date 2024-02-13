import numpy as np
import pytest

from epsilon_transformers.processes.zero_one_random import ZeroOneR

# TODO: Check for off by 1 error in the asserts
# TODO: Check that histogram distribution matches steady state distribution
# TODO: parameterize this for each process
# TODO: parameterize current_state_index

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