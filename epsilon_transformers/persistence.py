from abc import ABC
from io import BytesIO
import os
import pathlib
import boto3
import dotenv
import torch
from typing import TypeVar

TorchModule = TypeVar("TorchModule", torch.nn.modules.Module)
# TODO: Refactor tests
# TODO: Implement save_model overwrite protection
# TODO: Use TypeVar

class Persister(ABC):
    collection_location: pathlib.Path | str

    def save_model(self, model: TorchModule, num_tokens_trained: int):
        ...

    def load_model(self, model_class: TorchModule, object_name: str) -> TorchModule:
        ...

class LocalPersister(Persister):
    def __init__(self, collection_location: pathlib.Path):
        assert collection_location.is_dir()
        assert collection_location.exists()
        self.collection_location = collection_location        

    def save_model(self, model: TorchModule, num_tokens_trained: int):
        save_path = self.collection_location / f"{num_tokens_trained}.pt"
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

    def save_model(self, model: TorchModule, num_tokens_trained: int):
        buffer = BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        self.s3.upload_fileobj(buffer, self.collection_location, f'{num_tokens_trained}.pt')

    def load_model(self, model_class: TorchModule, object_name: str) -> TorchModule:
        download_buffer = BytesIO()
        self.s3.download_fileobj(self.collection_location, object_name, download_buffer)
        download_buffer.seek(0)
        return model_class.load_state_dict(torch.load(download_buffer))