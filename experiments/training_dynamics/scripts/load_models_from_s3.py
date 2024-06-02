import pathlib
import torch

from epsilon_transformers.persistence import S3Persister

def _load_and_save(name: str, stride: int, device: torch.device = torch.device('cpu'), overwrite_protection: bool = True):
    persister = S3Persister(collection_location=name)
    objects = persister.list_objects()
    sorted_model_token_num = sorted([int(x[:-3]) for x in objects if x[-3:] == '.pt'])
    strided_sorted_model_num_tokens = [x for i, x in enumerate(sorted_model_token_num) if i % stride == 0 or i == len(sorted_model_token_num) - 1]
    for num_tokens in strided_sorted_model_num_tokens:
        model = persister.load_model(object_name=f"{num_tokens}.pt", device=device)

        # CP from from epsilon_transformers.persistence.LocalPersister

        save_path = pathlib.Path(f"./experiments/models/{name}/{num_tokens}.pt")
        if save_path.exists() and overwrite_protection:
            print(f"Overwrite Protection: {save_path} already exists.")
            continue

        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    _load_and_save(name='rrxor', stride=1000)
