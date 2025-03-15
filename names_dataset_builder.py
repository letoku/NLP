import os
import json
import random
import torch

DATA_PATH = './data/'
SEPARATION_TOKEN = '.'


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

    def build(self) -> None:
        self._create_dataset_directory()
        self._build_alphabet()
        self._build_stoi_itos()
        self._save_stoi_itos()
        self._build_datasets()
        self._save_datasets()


    def _build_alphabet(self) -> None:
        concatenated_names = SEPARATION_TOKEN.join(self.names)
        alphabet_set = set(concatenated_names)

        self.alphabet = sorted(alphabet_set)

    def _build_stoi_itos(self) -> None:
        for i, l in enumerate(self.alphabet):
            self.stoi[l] = i
        self.itos = {i: l for l, i in self.stoi.items()}

    def _create_dataset_directory(self) -> None:
        self.dataset_dir = os.path.join(self.data_path, f'names__block_size_{self.block_size}')
        os.makedirs(self.dataset_dir)
        os.makedirs(os.path.join(self.dataset_dir, "train"))
        os.makedirs(os.path.join(self.dataset_dir, "val"))
        os.makedirs(os.path.join(self.dataset_dir, "test"))

    def _save_stoi_itos(self) -> None:
        with open(os.path.join(self.dataset_dir, "stoi.json"), "w") as f:
            json.dump(self.stoi, f, indent=4)
        with open(os.path.join(self.dataset_dir, "itos.json"), "w") as f:
            json.dump(self.itos, f, indent=4)

    @staticmethod
    def _split_names(names: list[str], train_frac:float=0.8, val_frac:float=0.1) -> (list[str], list[str], list[str]):
        assert train_frac + val_frac < 1.0
        names_shuffled = random.sample(names, len(names))

        train_size = int(len(names_shuffled) * train_frac)
        val_size = int(len(names_shuffled) * val_frac)

        train_data = names_shuffled[:train_size]
        val_data = names_shuffled[train_size:train_size + val_size]
        test_data = names_shuffled[train_size + val_size:]

        return train_data, val_data, test_data

    def _build_single_dataset(self, dataset: list[str]) -> (torch.Tensor, torch.Tensor):
        X = []
        y = []
        for name in dataset:
            context = self.block_size * [self.stoi[SEPARATION_TOKEN]]
            name_with_end = name + SEPARATION_TOKEN
            for char in name_with_end:
                X.append(context)
                y.append(self.stoi[char])
                context = context[1:] + [self.stoi[char]]

        X = torch.tensor(X)
        y = torch.tensor(y)

        return X, y

    def _build_datasets(self) -> None:
        train_data, val_data, test_data = self._split_names(self.names, self.train_frac, self.val_frac)
        self.X_train, self.y_train = self._build_single_dataset(train_data)
        self.X_val, self.y_val = self._build_single_dataset(val_data)
        self.X_test, self.y_test = self._build_single_dataset(test_data)

    def _save_single_dataset(self, X: torch.Tensor, y: torch.Tensor, name: str) -> None:
        X_dir = os.path.join(self.dataset_dir, name, "X.pt")
        y_dir = os.path.join(self.dataset_dir, name, "y.pt")
        torch.save(X, X_dir)
        torch.save(y, y_dir)

    def _save_datasets(self) -> None:
        self._save_single_dataset(self.X_train, self.y_train, 'train')
        self._save_single_dataset(self.X_val, self.y_val, 'val')
        self._save_single_dataset(self.X_test, self.y_test, 'test')