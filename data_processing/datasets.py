from typing import Tuple, Dict
import os
import json
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from abc import ABC, abstractmethod


TARGET_PADDING_VALUE = -100  # By default, in torch this value is ignored for loss calculations.


class TextDataset(Dataset, ABC):
    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val", "test"], "Split should be one of 'train', 'val', or 'test'"
        self.root = root
        self.data_path = os.path.join(root, split)
        self.transform = transform
        self.itos: Dict[int, str] = {}
        self.stoi: Dict[str, int] = {}
        self.size = 0

        with open(os.path.join(self.data_path, "data.pkl"), "rb") as f:
            self.data = pickle.load(f)
        self._load_stoi_itos()
        self._preprocess_data()


    def __len__(self):
        return self.size

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

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


class ContextWindowTextDataset(TextDataset):
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


class RNNTextDataset(TextDataset):
    def __init__(self, root, split="train", transform=None):
        super().__init__(root, split, transform)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.error_weights[idx]

    def _preprocess_data(self):
        max_len = max([len(sequence) for sequence in self.data])
        self.X = []
        self.y = []
        self.error_weights = []
        self.tokens_in_set = 0

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
            self.tokens_in_set += len(self.data[i])

        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)
        self.error_weights = torch.tensor(self.error_weights)
        self.size = len(self.X)

    def number_of_tokens_in_set(self) -> int:
        return self.tokens_in_set


class TorchRNNTextDataset(TextDataset):
    def __init__(self, root, split="train", transform=None):
        super().__init__(root, split, transform)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def _preprocess_data(self):
        self.X = []
        self.y = []
        self.tokens_in_set = 0

        for i in range(len(self.data)):
            input_seq = [0]
            output_seq = []
            for j in range(len(self.data[i])):
                output_seq.append(self.data[i][j])
                input_seq.append(self.data[i][j])
            input_seq = input_seq[:-1]
            self.X.append(torch.tensor(input_seq))
            self.y.append(torch.tensor(output_seq))
            self.tokens_in_set += len(self.data[i])

        self.size = len(self.X)

    def number_of_tokens_in_set(self) -> int:
        return self.tokens_in_set

    @staticmethod
    def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = zip(*batch)

        # Sort by descending length for pack_padded_sequence
        input_lengths = torch.tensor([len(seq) for seq in inputs])
        sorted_indices = torch.argsort(input_lengths, descending=True)

        sorted_inputs = [inputs[i] for i in sorted_indices]
        sorted_targets = [targets[i] for i in sorted_indices]
        sorted_lengths = input_lengths[sorted_indices]

        inputs_padded = pad_sequence(sorted_inputs, batch_first=True, padding_value=0)
        targets_padded = pad_sequence(sorted_targets, batch_first=True, padding_value=TARGET_PADDING_VALUE)

        return inputs_padded, targets_padded, sorted_lengths

