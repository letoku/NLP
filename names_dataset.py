import os
import json
import torch
from torch.utils.data import Dataset


class NamesDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val", "test"]
        self.root = root
        self.data_path = os.path.join(root, split)
        self.transform = transform
        self.X = torch.load(os.path.join(self.data_path, "X.pt"))
        self.y = torch.load(os.path.join(self.data_path, "y.pt"))
        self._load_stoi_itos()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    @staticmethod
    def _convert_keys_to_int(d):
        return {int(k): v for k, v in d.items()}

    def _load_stoi_itos(self):
        with open(os.path.join(self.root, "itos.json"), "r") as f:
            itos = json.load(f)
            self.itos = self._convert_keys_to_int(itos)
        with open(os.path.join(self.root, "stoi.json"), "r") as f:
            stoi = json.load(f)
