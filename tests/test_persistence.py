import boto3
import torch
from io import BytesIO
from dotenv import load_dotenv

# TODO: Implement this into the actual code base
# TODO: Create a PIBBSS S3 account
# TODO: Get Adam to move the relevant models into the new S3 account and buckets
# TODO: Write up the ReadMe stuff
# TODO: E2E training persistence example

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
    test_s3_create_and_delete_bucket()