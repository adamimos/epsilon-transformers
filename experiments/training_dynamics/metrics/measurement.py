import torch
from abc import ABC, abstractclassmethod, classmethod
import pathlib

# TODO: Test _tensor_fetcher to make sure it works when there isn't a .
# TODO: Change dataclass into pydantic??
# TODO: Create a config for running the sweep
# TODO: Create a save to disk function
# TODO: Test that property functions save to disk
# TODO: Add checkpoint_index to attn_circuit_measurements
# TODO: Write tests

class Measurement(ABC):
    model_name: str
    num_tokens_seen: int

    def _tensor_fetcher(self, property_path: str) -> torch.Tensor:
        properties = property_path.split('.')
        out = self
        for prop_name in properties:
            if hasattr(out, prop_name):
                out = getattr(out, prop_name)
            else:
                raise ValueError(f"Property '{property_path}' not found.")
        assert isinstance(out, torch.Tensor), '_tensor_fetcher() only works w/ tensors'
        return out

    def mean_reducer(self, property_path: str) -> float:
        tensor = self._tensor_fetcher(property_path=property_path)
        return torch.mean(tensor).item()

    def tensor_min(self, property_path: str) -> float:
        tensor = self._tensor_fetcher(property_path=property_path)
        flattened_tensor = torch.flatten(tensor)
        return torch.min(flattened_tensor).item()
        
    def tensor_max(self, property_path: str) -> float:
        tensor = self._tensor_fetcher(property_path=property_path)
        flattened_tensor = torch.flatten(tensor)
        return torch.max(flattened_tensor).item()

    def save_to_disk(self, outpath: pathlib.Path):
        raise NotImplementedError

    @classmethod @abstractclassmethod
    def from_model():
        ...