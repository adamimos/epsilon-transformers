from torch.utils.data import Dataset

from epsilon_transformers.processes.process import PROCESS_REGISTRY

# TODO: Change ProcessHistory so that we yield rather than create a full list
# TODO: Check for off by one errors in __getitem__

class ProcessDataset(Dataset):
    def __init__(self, process_name: str, total_samples: int, sequence_length: int):
        process = PROCESS_REGISTRY.get(process_name, None)
        if process is None:
            raise ValueError(f"{process_name} is not a recognized process")
        self.sequence_length = sequence_length
        self.process_history = process.generate_process_history(total_length=(total_samples * sequence_length) + 1)
       
    def __len__(self):
        return len(self.process_history) // self.sequence_length
    
    def __getitem__(self, idx):
        first_idx = idx * self.sequence_length
        snd_idx = (idx + 1) * self.sequence_length

        return self.process_history.symbols[first_idx: snd_idx], self.process_history[first_idx + 1 : snd_idx + 1]