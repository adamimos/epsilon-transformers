from epsilon_transformers.processes.zero_one_r import ZeroOneR

# TODO: Parameterize this test to work with all the different instantiations of the abstract class 
# TODO: Add relevant asserts for the sequence generations

def test_zero_one_r():
    process = ZeroOneR()
    assert process.num_states == 3
    assert process.vocab_len == 2
    assert process.is_unifilar()
    foo = process.generate_single_sequence(50, return_states=True)
    bar = process.generate_multiple_sequences(4, 5, return_states=True)
    print('huaah!')

def test_network_training():
    raise NotImplementedError

def test_analysis():
    raise NotImplementedError

def test_recurrent_subnetwork_msp_is_eps_machine():
    raise NotImplementedError

if __name__ == "__main__":
    test_zero_one_r()