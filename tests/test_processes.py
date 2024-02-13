import numpy as np
import pytest

from epsilon_transformers.processes.process import ProcessHistory
from epsilon_transformers.processes.zero_one_random import ZeroOneR

# TODO: Check for off by 1 error in the sample_emission asserts
# TODO: Check that histogram distribution matches steady state distribution in test_generate_single_sequence as well as multpiple_sequences

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

def test_generate_single_sequence():
    process = ZeroOneR()

    outs = process.generate_single_sequence(12)
    assert len(outs) == 12
    assert isinstance(outs, ProcessHistory)

def test_generate_multiple_sequences():
    process = ZeroOneR()

    outs = process.generate_multiple_sequences(12, 18)
    assert len(outs) == 12
    for seq in outs:
        assert len(seq) == 18
        assert isinstance(seq, ProcessHistory)