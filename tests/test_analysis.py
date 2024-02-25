import numpy as np
import pytest
from transformer_lens import HookedTransformer
import torch
from epsilon_transformers.analysis.activation_analysis import find_msp_subspace_in_residual_stream, generate_belief_state_and_activations

from epsilon_transformers.process.processes import ZeroOneR
from epsilon_transformers.training.configs import RawModelConfig

def test_msp_path_to_beliefs():
    process = ZeroOneR()
    msp = process.derive_mixed_state_presentation(depth=5)
    out = msp.path_to_beliefs(path=[0,1,1,0,1])
    assert np.array_equal(out, np.array([[1/3, 2/3, 0.        ],
       [0.        , 0.        , 1.        ],
       [1.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        ],
       [0.        , 0.        , 1.        ]]))
    
    with pytest.raises(AssertionError):
        out = msp.path_to_beliefs(path=[0,1,1,0,1,0,0,1,1])

    with pytest.raises(AssertionError):
        out = msp.path_to_beliefs(path=[1,1,1])


def test_generate_belief_state_and_activations():
    model_config = RawModelConfig(
        d_vocab=2,
        d_model=100,
        n_ctx=10,
        d_head=48,
        n_head=12,
        d_mlp=12,
        n_layers=2,
    )
    model = model_config.to_hooked_transformer(seed=13, device='cpu')
    process = ZeroOneR()

    belief_states, activations = generate_belief_state_and_activations(model=model, process=process, num_sequences=5)

    assert belief_states.shape[0] == activations.shape[0] == 5
    assert belief_states.shape[1] == activations.shape[1] == model.cfg.n_ctx
    assert belief_states.shape[2] == process.num_states
    assert activations.shape[2] == model.cfg.d_model

# TODO: Test find_msp_subspace_in_residual_stream (assert rank of predicted beliefs??)
def test_msp_subspace_in_residual_stream():
    model_config = RawModelConfig(
        d_vocab=2,
        d_model=100,
        n_ctx=10,
        d_head=48,
        n_head=12,
        d_mlp=12,
        n_layers=2,
    )
    model = model_config.to_hooked_transformer(seed=13, device='cpu')
    process = ZeroOneR()

    belief_states_reshaped, predicted_beliefs = find_msp_subspace_in_residual_stream(model=model, process=process, num_sequences=5)

    print(belief_states_reshaped.shape)
    print(predicted_beliefs.shape)
    
# TODO: Visualization tests
# TODO: E2E test w/ training
if __name__ == "__main__":
    test_msp_subspace_in_residual_stream()