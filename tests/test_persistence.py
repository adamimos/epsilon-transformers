import boto3
import torch
from io import BytesIO
from dotenv import load_dotenv

# TODO: Implement create a new bucket
# TODO: Implement this into the actual code base
# TODO: Add a delete junk into this test
# TODO: Create a PIBBSS S3 account
# TODO: Get Adam to move the relevant models into the new S3 account and buckets
# TODO: Write up the ReadMe stuff

def test_s3_persistence():
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

if __name__ == "__main__":
    test_s3_persistence()