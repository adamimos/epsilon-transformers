from torch.utils.data import DataLoader

from epsilon_transformers.process.dataset import ProcessDataset, process_dataset_collate_fn

def test_process_dataset():
    dataset = ProcessDataset('z1r', 10, 15)
    
    for data, label in dataset:
        assert len(data) == len(label) == 10
        assert data[1:] == label[:-1]

    dataset = ProcessDataset('z1r', 10, 16)
    dataloader = DataLoader(dataset=dataset, collate_fn=process_dataset_collate_fn, batch_size=2)

    for data, label in dataloader:
        assert len(data) == len(label) == 2  # Since batch_size is set to 2
        assert (data[:, 1:] == label[:, :-1]).all()
