import pathlib
import tempfile
import boto3
import pytest
import torch
from io import BytesIO
from dotenv import load_dotenv

from epsilon_transformers.persistence import LocalPersister, S3Persister

# TODO: Create a PIBBSS S3 account
# TODO: Get Adam to move the relevant models into the new S3 account and buckets
# TODO: E2E training persistence example

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
    raise NotImplementedError

def save_and_load_s3_model():
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
    s3.download_fileobj(bucket_name, 'model.pt', download_buffer, 85)

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
    loaded_model = persister.load_model(SimpleNN(), "85.pt")
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
    save_and_load_s3_model()