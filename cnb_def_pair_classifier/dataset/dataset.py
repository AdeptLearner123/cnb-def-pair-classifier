import torch
import json

class DefPairsDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        with open(path, "r") as file:
            self._instances = json.loads(file.read())
    
    def __getitem__(self, idx):
        return self._instances[idx]

    def __len__(self):
        return len(self._instances)