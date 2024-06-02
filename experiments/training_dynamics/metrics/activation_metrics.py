import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer
from typing import Literal, List, Dict, Optional
from dataclasses import dataclass
from datasets import load_dataset

from .measurement import Measurement

# TODO: Double check whether attn_eff_dim language is correct in the write up.
# TODO: Include mathematical description of attn_eff_dim in description
# TODO: Test in eff_dim (module name preservation, & expected values)

def prep_dataset(dataset_name: str, dataset_split: str, num_samples: int, max_token_len: int, model_name: str):
    dataset = load_dataset(dataset_name)
    _tokenizer = HookedTransformer.from_pretrained(model_name).tokenizer
    
    out = []
    for i in range(num_samples):
        text = dataset[dataset_split][i]['text']
        tokens = _tokenizer(text)['input_ids']
        out_str = _tokenizer.decode(tokens[:max_token_len])
        out.append(out_str)
    return out


Module = Literal['attn_out', 'mlp_out']

@dataclass
class Extracted(Measurement):
    model_name: str
    num_tokens_seen: int
    data: str
    logits: Optional[Float[torch.Tensor, "seq_pos vocab_len"]] = None
    activations: Optional[Dict[str, Float[torch.Tensor, "seq_pos d_model"]]] = None

    def from_model(
      data_str: str, 
      model: HookedTransformer, 
      extract_logits: bool, 
      extract_activations: List[Module]
    ) -> 'Extracted':
      if not extract_activations:
        logits = model(data_str, return_type="logits")
        return Extracted(
            model_name=model.cfg.model_name, 
            num_tokens_seen=model.cfg.checkpoint_value, 
            data=data_str, 
            logits=logits
        )
    
      logits, cache = model.run_with_cache(data_str, return_type="logits", loss_per_token=True, remove_batch_dim=True)
      activations = {key: activation for key, activation in cache.items() if any([x in key for x in extract_activations])}
      return Extracted(
          model_name=model.cfg.model_name, 
          num_tokens_seen=model.cfg.checkpoint_value, 
          data=data_str, 
          logits=logits if extract_logits else None,
          activations= activations
      )

def activation_effective_dimension(module_activations_dict: Dict[str, Float[torch.Tensor, 'seq_pos d_model']]) -> Dict[str, float]:
    with torch.no_grad():
      out = dict()
      for module_name, activation in module_activations_dict.items():
        eigenvalues: Float[torch.Tensor, "seq_pos"] = torch.linalg.eigvalsh(activation @ activation.T)
        activation_eff_dim: Float[torch.Tensor, ""] = eigenvalues.sum()**2 / (eigenvalues**2).sum()
        out[module_name] = activation_eff_dim.item()
      return out

def save_activations_as_jsonl(foo: List[Extracted]):
    raise NotImplementedError