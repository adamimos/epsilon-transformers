import torch
from transformer_lens import HookedTransformer, FactoredMatrix # type: ignore
from jaxtyping import Float
from dataclasses import dataclass
from typing import Literal, Optional, List

from .measurement import Measurement

# TODO: Test for Memory Leakages
# TODO: Change dataclass into pydantic??
# TODO: Write mathematical description of effective_functions

AttnCircuit = Literal['qk', 'ov']
Metric = Literal['effective_dimensionality', 'effective_rank']

@dataclass
class IndividualAttnCircuitMeasurement:
    circuit_type: AttnCircuit
    circuit: FactoredMatrix

    @property
    def effective_dimension(self) -> Float[torch.Tensor, "n_layers n_heads"]:
        # circuit: Float[torch.Tensor, "n_layers n_heads d_model d_model"]
        with torch.no_grad():
            singular_values: Float[torch.Tensor, "n_layers n_heads d_head"] = self.circuit.svd()[1]
            return singular_values.sum(dim=-1)**2 / (singular_values**2).sum(dim=-1)

    @property
    def effective_rank(self) -> Float[torch.Tensor, "n_layers n_heads"]:
        # circuit: Float[torch.Tensor, "n_layers n_heads d_model d_model"]
        with torch.no_grad():
            singular_values: Float[torch.Tensor, "n_layers n_heads d_head"] = self.circuit.svd()[1]
            singular_values_normalized: Float[torch.Tensor, "n_layers n_heads d_head"] = singular_values / singular_values.sum(dim=-1, keepdim=True)
            singular_value_entropy: Float[torch.Tensor, "n_layers n_heads"] = (-singular_values_normalized * torch.log(singular_values_normalized)).sum(dim=-1)  # TODO: Do we need a + EPS here? 
            return torch.exp(singular_value_entropy)

@dataclass
class AttnCircuitsMeasurements(Measurement):
    model_name: str
    num_tokens_seen: int
    qk: Optional[IndividualAttnCircuitMeasurement]
    ov: Optional[IndividualAttnCircuitMeasurement]

    def __post_init__(self):
        assert self.qk is not None or self.ov is not None, ''

    @classmethod
    def from_model(cls, model: HookedTransformer, circuits: List[AttnCircuit]) -> 'AttnCircuitsMeasurements':
        assert circuits, 'circuits cannot be an empty list'
        return AttnCircuitsMeasurements(
            model_name=model.cfg.model_name,
            num_tokens_seen=model.cfg.checkpoint_value,
            qk=IndividualAttnCircuitMeasurement(circuit_type='qk', circuit=model.QK) if 'qk' in circuits else None,
            ov=IndividualAttnCircuitMeasurement(circuit_type='ov', circuit=model.OV) if 'ov' in circuits else None,
        )

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("pythia-160m", checkpoint_index=-1, device='cpu')
    measurement = AttnCircuitsMeasurements.from_model(model, ['qk', 'ov'])
    measurement.save_to_disk()