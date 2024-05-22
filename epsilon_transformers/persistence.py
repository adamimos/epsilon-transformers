from abc import ABC
from dataclasses import dataclass
from io import BytesIO
import os
import pathlib
import re
from botocore.exceptions import ClientError
import boto3
import dotenv
import torch
from typing import Dict, List, OrderedDict, Tuple, TypeVar
import json
import pandas as pd

from epsilon_transformers.training.configs.model_configs import RawModelConfig

class HackyPersister:
    dir_path: pathlib.Path

    def __init__(self, dir_path: pathlib.Path):
        self.dir_path = dir_path

    def get_model_checkpoints(self):
        pt_files = [file for file in self.dir_path.glob('*.pt') if file.is_file()]
        int_list = [int(path.stem) for path in pt_files]
        sorted_int_list = sorted(int_list)
        return sorted_int_list

    def load_training_config(self):
        with open(self.dir_path / 'train_config.json', 'r') as json_file:
            return json.load(json_file)
    
    
    def load_model(self, ckpt_num: int, device: str):

        train_config = self.load_training_config()
        
        if train_config is None:
            # print("No train_config.json found, inferring from state_dict")
            # config = _state_dict_to_model_config(state_dict=state_dict)
            raise ValueError("traing config not found")
        else:
            # TODO: refactor this
            required_fields = ['d_vocab', 'd_model', 'n_ctx', 'd_head', 'n_heads', 'n_layers']
            config_dict = {k: v for k, v in train_config.items() if k in required_fields}
            config_dict['d_mlp'] = 4 * config_dict['d_model']
            # change key n_heads to n_head
            config_dict['n_head'] = config_dict.pop('n_heads')
            config = RawModelConfig(**config_dict)

        model = config.to_hooked_transformer(device=device)
        model.load_state_dict(state_dict=torch.load(self.dir_path / f"{ckpt_num}.pt"))
        
        return model


    def load_final_model(self):
        checkpoint_filenames = self.get_model_checkpoints()
        print(f"Checkpoints found: {len(checkpoint_filenames)}")
        model = self.load_model(ckpt_num=checkpoint_filenames[-1], device='cpu')
        print(f"Last checkpoint: {checkpoint_filenames[-1]}")
        return model


TorchModule = TypeVar("TorchModule", bound=torch.nn.modules.Module)

# TODO: Add "check for versioning" in S3
# TODO: Persist list of buckets on init
# TODO: When inferring correct config, log a warning to the user about n_ctx

# TODO: Create set collection_location
# TODO: Make all save_model functions async
# TODO: Create save config & implement it in the training loop
# TODO: Create query commit hash and add it to the save_config method

# TODO: Change _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY to nested dict
# TODO: Clean up _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY to have smaller num of keys
# TODO: Add a check to _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY to make sure every key is visited (??)
# TODO: Generalize _state_dict_to_model_config into it's own base class so that it can handle different kinds of model classes

class Persister(ABC):
    collection_location: pathlib.Path | str

    def save_model(self, model: TorchModule, num_tokens_trained: int):
        ...

    def load_model(self, model_class: TorchModule, object_name: str) -> TorchModule:
        ...
    
    def _save_overwrite_protection(self, object_name: pathlib.Path | str):
        ...

class LocalPersister(Persister):
    def __init__(self, collection_location: pathlib.Path):
        assert collection_location.is_dir()
        assert collection_location.exists()
        self.collection_location = collection_location        

    def _save_overwrite_protection(self, object_name: pathlib.Path):
        if object_name.exists():
            raise ValueError(f"Overwrite Protection: {object_name} already exists.")

    def save_model(self, model: TorchModule, num_tokens_trained: int):
        save_path: pathlib.Path = self.collection_location / f"{num_tokens_trained}.pt"
        self._save_overwrite_protection(object_name=save_path)

        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)

    def load_model(self, model: TorchModule, object_name: str) -> TorchModule:
        state_dict = torch.load(self.collection_location / object_name)
        model.load_state_dict(state_dict=state_dict)
        return model

class S3Persister(Persister):
    def __init__(self, collection_location: str):
        dotenv.load_dotenv()
        assert os.environ.get('AWS_ACCESS_KEY_ID') is not None
        assert os.environ.get('AWS_SECRET_ACCESS_KEY') is not None
        
        self.s3 = boto3.client('s3')
        buckets = [x['Name'] for x in self.s3.list_buckets()['Buckets']]
        if collection_location not in buckets:
            raise ValueError(f"{collection_location} is not an existing bucket. Either use one of the existing buckets or create a new bucket")
        self.collection_location = collection_location

    def _save_overwrite_protection(self, object_name: str):
        try:
            self.s3.head_object(Bucket=self.collection_location, Key=object_name)
            raise ValueError(f"Overwrite Protection: {self.collection_location}/{object_name} already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                pass
            else:
                raise ValueError(f"Expected 404 from empty object, received {e}")

    def save_model(self, model: TorchModule, num_tokens_trained: int):
        object_name = f'{num_tokens_trained}.pt'
        self._save_overwrite_protection(object_name=object_name)
        
        print(f"Saving model as {object_name} in bucket {self.collection_location}")
        
        buffer = BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        self.s3.upload_fileobj(buffer, self.collection_location, object_name)

    def load_csv(self, object_name: str) -> str:
        download_buffer = BytesIO()
        self.s3.download_fileobj(self.collection_location, object_name, download_buffer)
        download_buffer.seek(0)
        return pd.read_csv(download_buffer)

    def load_model(self, object_name: str, device: str) -> TorchModule:
        download_buffer = BytesIO()
        self.s3.download_fileobj(self.collection_location, object_name, download_buffer)
        download_buffer.seek(0)
        state_dict = torch.load(download_buffer)

        train_config = self.load_json('train_config.json')
        if train_config is not None:
            # TODO: refactor this
            required_fields = ['d_vocab', 'd_model', 'n_ctx', 'd_head', 'n_heads', 'n_layers']
            config_dict = {k: v for k, v in train_config.items() if k in required_fields}
            config_dict['d_mlp'] = 4 * config_dict['d_model']
            # change key n_heads to n_head
            config_dict['n_head'] = config_dict.pop('n_heads')
            config = RawModelConfig(**config_dict)
        else:
            print("No train_config.json found, inferring from state_dict")
            config = _state_dict_to_model_config(state_dict=state_dict)

        model = config.to_hooked_transformer(device=device)
        model.load_state_dict(state_dict=state_dict)
        return model
    
    def list_objects(self) -> List[str]:
        objects = []
        continuation_token = None

        while True:
            kwargs = {'Bucket': self.collection_location}
            if continuation_token:
                kwargs['ContinuationToken'] = continuation_token

            response = self.s3.list_objects_v2(**kwargs)
            contents = response.get('Contents', [])
            objects.extend([obj['Key'] for obj in contents])

            if 'NextContinuationToken' in response:
                continuation_token = response['NextContinuationToken']
            else:
                break

        return objects
    
    def load_json(self, object_name: str) -> Dict:
        try:
            json_str = self.load_object(object_name)
            return json.loads(json_str)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            else:
                raise e
    
    def load_object(self, object_name: str) -> str:
        download_buffer = BytesIO()
        self.s3.download_fileobj(self.collection_location, object_name, download_buffer)
        download_buffer.seek(0)
        return download_buffer.read().decode('utf-8')

def _state_dict_to_model_config(state_dict: OrderedDict, n_ctx: int = 10) -> RawModelConfig:
    _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY: Dict[str, List[Tuple[str, int]]] = {
        r"embed\.W_E": [('d_vocab', 0), ('d_model', 1)],
        r"pos_embed\.W_pos": [],
        r"blocks\.\d+\.ln\d+\.(w|b)": [],
        r"blocks\.\d+\.attn\.W_Q": [('n_head', 0), ('d_head', 2)],
        r"blocks\.\d+\.attn\.b_Q": [],
        r"blocks\.\d+\.attn\.W_K": [],
        r"blocks\.\d+\.attn\.b_K": [],
        r"blocks\.\d+\.attn\.W_O": [],
        r"blocks\.\d+\.attn\.b_O": [],
        r"blocks\.\d+\.attn\.W_V": [],
        r"blocks\.\d+\.attn\.b_V": [],
        r"blocks\.\d+\.attn\.mask": [],
        r"blocks\.\d+\.attn\.IGNORE": [],
        r"blocks\.\d+\.mlp\.W_in": [('d_mlp', 1)],
        r"blocks\.\d+\.mlp\.b_in": [],
        r"blocks\.\d+\.mlp\.W_out": [],
        r"blocks\.\d+\.mlp\.b_out": [],
        r"ln_final\.(w|b)": [],
        r"unembed\.(W_U|b_U)": []
    }
    def _extract_true_key(dictionary: Dict[str, bool]) -> str:
        out = []
        for key, value in dictionary.items():
            if value:
                out.append(key)
        assert len(out) == 1,  f"{out} does not fit one of the expected module regexs: {_HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY}"
        return out[0]
    def _extract_n_layers(state_dict: OrderedDict) -> int:
        highest_block_idx = None
        for key in state_dict.keys():
            if not bool(re.match(r"blocks\.\d+\.", key)):
                continue
            local_block_idx = int(re.search(r'\d+', key).group())
            if highest_block_idx is None:
                highest_block_idx = local_block_idx
            elif local_block_idx > highest_block_idx:
                highest_block_idx = local_block_idx
        return highest_block_idx + 1

    param_dict = dict(d_vocab=None, d_model=None, n_ctx=n_ctx, d_head=None, n_head=None, d_mlp=None, n_layers=_extract_n_layers(state_dict=state_dict))
    for module_name, module in state_dict.items():
        regex_dict = {pattern: bool(re.match(pattern, module_name)) for pattern in _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY.keys()}
        pattern = _extract_true_key(regex_dict)
        for key, dim in _HOOKED_TRANSFORMER_MODULE_REGEXES_REGISTRY[pattern]:
            if param_dict[key] is None:
                param_dict[key] = module.size()[dim]
    assert all([value is not None for value in param_dict.values()])
    return RawModelConfig(**param_dict)

if __name__ == "__main__":
    persister = S3Persister(collection_location="mess3-param-change")
    model_class = None
    from transformer_lens import HookedTransformer
    model: HookedTransformer = persister.load_model(device='cpu', object_name='4800000.pt')