import pathlib
import tempfile
import boto3
import pytest
import torch
from io import BytesIO
from dotenv import load_dotenv

from epsilon_transformers.persistence import LocalPersister, S3Persister
from epsilon_transformers.training.configs.training_configs import LoggingConfig, OptimizerConfig, PersistanceConfig, ProcessDatasetConfig, TrainConfig
from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.training.train import train_model

# TODO: Add e2e training check for expected saved models
# TODO: Refactor the tests to use SimpleNN as fixture and random init the params

# TODO: Put slow tags on all s3 tests
# TODO: Write tests for local save_model overwrite protection
# TODO: Add a reset to the bucket state before running all the tests
# TODO: Move test non existing bucket into it's own test

def test_e2e_training():
    bucket_name = 'lucas-testing-rrxor-s3-training'
    s3 = boto3.client('s3')
    s3.create_bucket(Bucket=bucket_name)
    
    model_config = RawModelConfig(
            d_vocab=2,
            d_model=100,
            n_ctx=10,
            d_head=48,
            n_head=12,
            d_mlp=12,
            n_layers=2,
        )
    optimizer_config = OptimizerConfig(
        optimizer_type='adam',
        learning_rate=1.06e-4,
        weight_decay=0.8
    )

    dataset_config = ProcessDatasetConfig(
        process='rrxor',
        batch_size=5,
        num_tokens=500,
        test_split=0.15
    )

    persistance_config = PersistanceConfig(
        location='s3',
        collection_location= 'lucas-testing-rrxor-s3-training',
        checkpoint_every_n_tokens=100
    )

    mock_config = TrainConfig(
        model=model_config,
        optimizer=optimizer_config,
        dataset=dataset_config,
        persistance=persistance_config,
        logging=LoggingConfig(project_name='lucas-testing-rrxor-s3-training', wandb=False),
        verbose=True,
        seed=1337
    )
    train_model(mock_config)

    s3.delete_bucket(Bucket=bucket_name)



def test_s3_save_model_overwrite_protection():
    # Define a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)
        
    # Create an instance of the neural network
    network = SimpleNN()

    # Serialize the network to bytes
    buffer = BytesIO()
    torch.save(network.state_dict(), buffer)

    # Upload the serialized network to the bucket
    load_dotenv()
    s3 = boto3.client('s3')
    bucket_name = 'lucas-getting-started-with-s3-demo'
    
    buffer.seek(0)  # Reset buffer position to the beginning
    s3.upload_fileobj(buffer, bucket_name, '45.pt')
    
    persister = S3Persister(collection_location=bucket_name)

    with pytest.raises(ValueError):
        persister.save_model(network, 45)

def load_local_model():
    # Define a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Create an instance of the neural network
    model = SimpleNN()

    with tempfile.TemporaryDirectory() as temp_dir:
        model_filepath = pathlib.Path(temp_dir) / "model.pt"
        torch.save(model.state_dict(), model_filepath)

        persister = LocalPersister(collection_location=pathlib.Path(temp_dir))
        loaded_model = persister.load_model(model=SimpleNN(), object_name="model.pt")

    assert torch.all(torch.eq(loaded_model.state_dict()['fc.weight'], model.state_dict()['fc.weight']))
    assert torch.all(torch.eq(loaded_model.state_dict()['fc.bias'], model.state_dict()['fc.bias']))

def save_local_model():
    # Define a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Create an instance of the neural network
    model = SimpleNN()

    with tempfile.TemporaryDirectory() as temp_dir:
        persister = LocalPersister(collection_location=pathlib.Path(temp_dir))
        num_tokens = 45
        
        persister.save_model(model, num_tokens)

        loaded_model = SimpleNN()
        loaded_model_dict = torch.load(pathlib.Path(temp_dir) / f"{num_tokens}.pt")
        loaded_model.load_state_dict(loaded_model_dict)
    assert torch.all(torch.eq(loaded_model.state_dict()['fc.weight'], model.state_dict()['fc.weight']))
    assert torch.all(torch.eq(loaded_model.state_dict()['fc.bias'], model.state_dict()['fc.bias']))


def test_save_and_load_s3_model():
    bucket_name = 'lucas-new-test-bucket-003'
    with pytest.raises(ValueError):
        S3Persister(collection_location=bucket_name)
    
    # Create mock bucket
    s3 = boto3.client('s3')
    s3.create_bucket(Bucket=bucket_name)
    
    persister = S3Persister(collection_location=bucket_name)

    # mock nn
    # TODO: Mock needs to be randomly initialized
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    model = SimpleNN()
    
    # test save
    persister.save_model(model, 85)

    download_buffer = BytesIO()
    s3.download_fileobj(bucket_name, '85.pt', download_buffer)

    # Load the downloaded network
    downloaded_model = SimpleNN()
    download_buffer.seek(0)  # Reset download buffer position to the beginning
    downloaded_model.load_state_dict(torch.load(download_buffer))

    # Assert that the downloaded network is the same as the original one
    assert torch.all(torch.eq(model.state_dict()['fc.weight'], downloaded_model.state_dict()['fc.weight']))
    assert torch.all(torch.eq(model.state_dict()['fc.bias'], downloaded_model.state_dict()['fc.bias']))

    # Test save overwrite protection
    with pytest.raises(ValueError):
        persister.save_model(model, 85)

    # Test load
    loaded_model = SimpleNN()
    persister.load_model(loaded_model, "85.pt")
    assert torch.all(torch.eq(model.state_dict()['fc.weight'], loaded_model.state_dict()['fc.weight']))
    assert torch.all(torch.eq(model.state_dict()['fc.bias'], loaded_model.state_dict()['fc.bias']))

    # Delete mock bucket
    s3.delete_object(Bucket=bucket_name, Key='85.pt')
    s3.delete_bucket(Bucket=bucket_name)

def test_s3_persistence_put_and_retrieve_object_from_bucket():
    # Define a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Create an instance of the neural network
    network = SimpleNN()

    # Serialize the network to bytes
    buffer = BytesIO()
    torch.save(network.state_dict(), buffer)

    # Upload the serialized network to the bucket
    load_dotenv()
    s3 = boto3.client('s3')
    bucket_name = 'lucas-getting-started-with-s3-demo'
    buffer.seek(0)  # Reset buffer position to the beginning
    s3.upload_fileobj(buffer, bucket_name, 'model.pt')

    # Download the serialized network from the bucket
    download_buffer = BytesIO()
    s3.download_fileobj(bucket_name, 'model.pt', download_buffer)

    # Load the downloaded network
    downloaded_network = SimpleNN()
    download_buffer.seek(0)  # Reset download buffer position to the beginning
    downloaded_network.load_state_dict(torch.load(download_buffer))

    # Assert that the downloaded network is the same as the original one
    assert torch.all(torch.eq(network.state_dict()['fc.weight'], downloaded_network.state_dict()['fc.weight']))
    assert torch.all(torch.eq(network.state_dict()['fc.bias'], downloaded_network.state_dict()['fc.bias']))

    s3.delete_object(Bucket=bucket_name, Key='model.pt')


def test_s3_create_and_delete_bucket():
    # Define a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Create an instance of the neural network
    network = SimpleNN()

    # Serialize the network to bytes
    buffer = BytesIO()
    torch.save(network.state_dict(), buffer)

    # Create a new S3 bucket
    s3 = boto3.client('s3')
    bucket_name = 'lucas-new-test-bucket-003'
    s3.create_bucket(Bucket=bucket_name)

    # Upload the serialized network to the bucket
    buffer.seek(0)  # Reset buffer position to the beginning
    s3.upload_fileobj(buffer, bucket_name, 'model.pt')

    # Download the serialized network from the bucket
    download_buffer = BytesIO()
    s3.download_fileobj(bucket_name, 'model.pt', download_buffer)

    # Load the downloaded network
    downloaded_network = SimpleNN()
    download_buffer.seek(0)  # Reset download buffer position to the beginning
    downloaded_network.load_state_dict(torch.load(download_buffer))

    # Assert that the downloaded network is the same as the original one
    assert torch.all(torch.eq(network.state_dict()['fc.weight'], downloaded_network.state_dict()['fc.weight']))
    assert torch.all(torch.eq(network.state_dict()['fc.bias'], downloaded_network.state_dict()['fc.bias']))

    # Clean up: delete the file and the bucket
    s3.delete_object(Bucket=bucket_name, Key='model.pt')
    s3.delete_bucket(Bucket=bucket_name)

    # Assert that the bucket was deleted
    response = s3.list_buckets()
    bucket_names = [bucket['Name'] for bucket in response['Buckets']]
    assert bucket_name not in bucket_names, f"Bucket {bucket_name} was not deleted"

if __name__ == "__main__":
    test_e2e_training()