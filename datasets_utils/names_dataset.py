import os
import json
import pickle
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class NamesDataset(Dataset, ABC):
    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val", "test"]
        self.root = root
        self.data_path = os.path.join(root, split)
        self.transform = transform
        with open(os.path.join(self.data_path, "data.pkl"), "rb") as f:
            self.data = pickle.load(f)
        self._load_stoi_itos()
        self.size = 0
        self._preprocess_data()

    def __len__(self):
        return self.size

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @staticmethod
    def _convert_keys_to_int(d):
        return {int(k): v for k, v in d.items()}

    def _load_stoi_itos(self):
        with open(os.path.join(self.root, "itos.json"), "r") as f:
            itos = json.load(f)
            self.itos = self._convert_keys_to_int(itos)
        with open(os.path.join(self.root, "stoi.json"), "r") as f:
            self.stoi = json.load(f)

    @abstractmethod
    def _preprocess_data(self):
        pass


class ContextWindowNamesDataset(NamesDataset):
    def __init__(self, root, context_size: int, split="train", transform=None):
        self.context_size = context_size
        super().__init__(root, split, transform)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def _preprocess_data(self):
        self.X = []
        self.y = []
        for i in range(len(self.data)):
            context = [0] * self.context_size
            for j in range(len(self.data[i])):
                self.X.append(context)
                self.y.append(self.data[i][j])
                context = context[1:] + [self.data[i][j]]

        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)
        self.size = len(self.X)


class RNNNamesDataset(NamesDataset):
    def __init__(self, root, split="train", transform=None):
        super().__init__(root, split, transform)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.error_weights[idx]

    def _preprocess_data(self):
        max_len = max([len(sequence) for sequence in self.data])
        self.X = []
        self.y = []
        self.error_weights = []
        for i in range(len(self.data)):
            input_seq = [0]
            output_seq = []
            error_weights = []
            for j in range(max_len):
                if j < len(self.data[i]):
                    output_seq.append(self.data[i][j])
                    error_weights.append(1)
                    input_seq.append(self.data[i][j])
                else:
                    output_seq.append(0)
                    input_seq.append(0)
                    error_weights.append(0)
            input_seq = input_seq[:-1]
            self.X.append(input_seq)
            self.y.append(output_seq)
            self.error_weights.append(error_weights)
            self.size += len(self.data[i])

        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)
        self.error_weights = torch.tensor(self.error_weights)
