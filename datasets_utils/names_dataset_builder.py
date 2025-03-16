import os
import json
import random
import pickle


SEPARATION_TOKEN = '.'


class NamesDatasetBuilder:
    def __init__(self, names: list[str], data_path: str,
                 train_frac: float = 0.8,
                 val_frac: float = 0.1):
        self.names = names
        self.data_path = data_path
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
        self.dataset_dir = os.path.join(self.data_path, f'names_dataset')
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

    def _build_single_dataset(self, dataset: list[str]) -> list[list[int]]:
        data = []
        for name in dataset:
            name_entry = []
            name_with_end = name + SEPARATION_TOKEN
            for char in name_with_end:
                name_entry.append(self.stoi[char])
            data.append(name_entry)

        return data

    def _build_datasets(self) -> None:
        train_data, val_data, test_data = self._split_names(self.names, self.train_frac, self.val_frac)
        self.train = self._build_single_dataset(train_data)
        self.val = self._build_single_dataset(val_data)
        self.test  = self._build_single_dataset(test_data)

    def _save_single_dataset(self, data: list[list[int]], name: str) -> None:
        data_dir = os.path.join(self.dataset_dir, name, "data.pkl")
        with open(data_dir, "wb") as f:
            pickle.dump(data, f)

    def _save_datasets(self) -> None:
        self._save_single_dataset(self.train, 'train')
        self._save_single_dataset(self.val, 'val')
        self._save_single_dataset(self.test, 'test')