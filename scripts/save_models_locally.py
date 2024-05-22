import json
import pathlib
from typing import List
from dataclasses import dataclass

import torch
from epsilon_transformers.persistence import S3Persister

def get_model_checkpoints(persister: S3Persister):
    filenames = persister.list_objects()
    filenames_pt = [x for x in filenames if ".pt" in x]
    filenames_pt.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return filenames_pt

# TODO: Add train_config
@dataclass
class PersisterAndCheckpoints:
    persister: S3Persister
    checkpoints: List[int]

    def save_model_local(self, dir_path: pathlib.Path):
        dir_path.mkdir(parents=True, exist_ok=True)

        for ckpt in self.checkpoints:
            model = self.persister.load_model(f"{ckpt}.pt", device='cpu')
            save_path: pathlib.Path = dir_path / f"{ckpt}.pt"

            print(f"Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)

    def save_local_train_config(self, dir_path: pathlib.Path):
        dir_path.mkdir(parents=True, exist_ok=True)

        config = self.persister.load_json(object_name="train_config.json")
        
        with open(dir_path / 'train_config.json', 'w') as json_file:
            json.dump(config, json_file)

if __name__ == "__main__":
    mess3 = PersisterAndCheckpoints(
    persister = S3Persister(collection_location="mess3-0.05-0.85-longrun"),
    checkpoints = [6400, 64000, 640000, 915200, 3187200, 629209600])

    # mess3.save_model_local(pathlib.Path("./examples/models/mess3"))
    mess3.save_local_train_config(pathlib.Path("./examples/models/mess3"))

    rrxor_persister = S3Persister(collection_location="rrxor")
    print()
# rrxor is bull, come back and fix it
    # rrxor = PersisterAndCheckpoints(
    # persister = rrxor_persister,
    # checkpoints = [get_model_checkpoints(rrxor_persister)[-1]])

    # mess3.save_model_local(pathlib.Path("./examples/models/rrxor"))
