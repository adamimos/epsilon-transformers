import pathlib
import torch
from typing import Iterable, List
from transformer_lens import HookedTransformer

from epsilon_transformers.persistence import state_dict_to_model_config

def _model_iterator_factory(dir: pathlib.Path, device: torch.device = torch.device('cpu')) -> Iterable[HookedTransformer]:
    assert dir.is_dir()
    assert dir.exists()

    files = [file for file in dir.glob('*.pt') if file.stem.isnumeric()]
    tokens_trained = sorted([int(str(x.stem)) for x in files if str(x)[-3:] == '.pt'])
    files_sorted = [f"{tokens}.pt" for tokens in tokens_trained]

    def _iterator(files: List[str]):
        for file in files:
            state_dict = torch.load(dir / file)
            config = state_dict_to_model_config(state_dict=state_dict)
            model = config.to_hooked_transformer(device=device)
            model.load_state_dict(state_dict=state_dict)
            yield model

    return _iterator(files=files_sorted)

if __name__ == "__main__":
    model_iterator = _model_iterator_factory(dir=pathlib.Path('./experiments/models/rrxor'))
    foo = next(model_iterator)
    print()