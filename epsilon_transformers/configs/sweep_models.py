from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Union, Optional, Any, Dict

class ListParameter(BaseModel):
    values: Optional[List[Union[int, float, str]]] = None

class DirectParameter(BaseModel):
    value: Optional[Union[int, float, str, bool]] = None

    # Adding a root validator to automatically extract value if it's a dict
    @root_validator(pre=True)
    def extract_value(cls, values):
        if 'value' in values:
            return {'value': values['value']}
        return values

class Metric(BaseModel):
    goal: str
    name: str


class Parameters(BaseModel):
    d_model: ListParameter
    d_head: ListParameter
    n_layers: ListParameter
    n_ctx: DirectParameter
    n_heads: ListParameter
    d_vocab: DirectParameter
    act_fn: DirectParameter
    use_attn_scale: DirectParameter
    normalization_type: DirectParameter
    attention_dir: DirectParameter
    attn_only: DirectParameter
    seed: DirectParameter
    init_weights: DirectParameter
    batch_size: ListParameter
    num_epochs: DirectParameter
    learning_rate: ListParameter
    weight_decay: ListParameter
    optimizer: ListParameter


class SweepConfig(BaseModel):
    method: str
    metric: Metric
    sweep_name: str
    process: str
    parameters: Parameters
