from typing import Optional
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from epsilon_transformers.training.configs.base_config import Config


class RawModelConfig(Config):
    d_vocab: int
    d_model: int
    n_ctx: int
    d_head: int
    n_head: int
    d_mlp: int
    n_layers: int

    def to_hooked_transformer(
        self, device: torch.device, seed: Optional[int] = None
    ) -> HookedTransformer:
        config = HookedTransformerConfig(
            d_model=self.d_model,
            d_head=self.d_head,
            n_layers=self.n_layers,
            n_ctx=self.n_ctx,
            n_heads=self.n_head,
            d_mlp=self.d_mlp,
            d_vocab=self.d_vocab,
            seed=seed,
            device=device,
            act_fn="relu",
        )
        return HookedTransformer(config)

