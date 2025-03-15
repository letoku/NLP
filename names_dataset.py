import os
import json
import torch

DATA_PATH = './data/'

class NamesDatasetBuilder:
    def __init__(self, names: list[str], block_size: int = 3, data_path: str = DATA_PATH,
                 train_frac: float = 0.8,
                 val_frac: float = 0.1):
        self.names = names
        self.data_path = data_path
        self.block_size = block_size
        self.train_frac = train_frac
        self.val_frac = val_frac

        self.alphabet = []
        self.stoi = {}
        self.itos = {}

    def build(self):
        self._create_dataset_directory()
        self._build_alphabet()
        self._build_stoi_itos()
        self._save_stoi_itos()


    def _build_alphabet(self) -> None:
        concatenated_names = '.'.join(self.names)
        alphabet_set = set(concatenated_names)

        self.alphabet = sorted(alphabet_set)

    def _build_stoi_itos(self) -> None:
        for i, l in enumerate(self.alphabet):
            self.stoi[l] = i
        self.itos = {i: l for l, i in self.stoi.items()}

    def _create_dataset_directory(self):
        self.dataset_dir = os.path.join(self.data_path, f'names__block_size_{self.block_size}')
        os.makedirs(self.dataset_dir)
        os.makedirs(os.path.join(self.dataset_dir, "train"))
        os.makedirs(os.path.join(self.dataset_dir, "val"))
        os.makedirs(os.path.join(self.dataset_dir, "test"))

    def _save_stoi_itos(self):
        with open(os.path.join(self.dataset_dir, "stoi.json"), "w") as f:
            json.dump(self.stoi, f, indent=4)
        with open(os.path.join(self.dataset_dir, "itos.json"), "w") as f:
            json.dump(self.itos, f, indent=4)
