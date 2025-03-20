from abc import abstractmethod, ABC
from typing import List
import os
import json
import pickle
import nltk

from ._utils import _split_dataset, _ensure_tokenizer_is_downloaded


class DatasetBuilder(ABC):
    def __init__(self, dataset_name: str, data_path: str,
                 separation_token: str,
                 train_frac: float = 0.8,
                 val_frac: float = 0.1):
        self.name = dataset_name
        self.data_path = data_path
        self.dataset_dir: str | None = None

        self.train_frac = train_frac
        self.val_frac = val_frac

        self.train = None
        self.val = None
        self.test = None

        self.separation_token = separation_token
        self.alphabet = []
        self.stoi = {}    # map from strings to integers
        self.itos = {}    # inverse map

    def build(self) -> None:
        self._create_dataset_directory()
        self._build_alphabet()
        self._build_stoi_itos()
        self._save_stoi_itos()
        self._build_datasets()
        self._save_datasets()

    @abstractmethod
    def _build_alphabet(self) -> None:
        pass

    def _build_stoi_itos(self) -> None:
        for i, l in enumerate(self.alphabet):
            self.stoi[l] = i
        self.itos = {i: l for l, i in self.stoi.items()}

    def _save_stoi_itos(self) -> None:
        with open(os.path.join(self.dataset_dir, "stoi.json"), "w") as f:
            json.dump(self.stoi, f, indent=4)
        with open(os.path.join(self.dataset_dir, "itos.json"), "w") as f:
            json.dump(self.itos, f, indent=4)

    @abstractmethod
    def _build_datasets(self) -> None:
        pass

    def _save_datasets(self) -> None:
        self._save_single_dataset(self.train, 'train')
        self._save_single_dataset(self.val, 'val')
        self._save_single_dataset(self.test, 'test')

    def _save_single_dataset(self, data: List[List[int]], name: str) -> None:
        data_dir = os.path.join(self.dataset_dir, name, "data.pkl")
        with open(data_dir, "wb") as f:
            pickle.dump(data, f)

    def _create_dataset_directory(self) -> None:
        self.dataset_dir = os.path.join(self.data_path, self.name)
        os.makedirs(self.dataset_dir)
        os.makedirs(os.path.join(self.dataset_dir, "train"))
        os.makedirs(os.path.join(self.dataset_dir, "val"))
        os.makedirs(os.path.join(self.dataset_dir, "test"))


class NamesDatasetBuilder(DatasetBuilder):
    def __init__(self, dataset_name: str, names: List[str], data_path: str, separation_token: str, train_frac: float = 0.8,
                 val_frac: float = 0.1):
        super().__init__(
            dataset_name=dataset_name,
            data_path=data_path,
            separation_token=separation_token,
            train_frac=train_frac,
            val_frac=val_frac
        )
        self.names = names

    def _build_alphabet(self) -> None:
        concatenated_names = self.separation_token.join(self.names)
        alphabet_set = set(concatenated_names)

        self.alphabet = sorted(alphabet_set)

    def _build_single_dataset(self, dataset: List[str]) -> List[List[int]]:
        data = []
        for name in dataset:
            name_encodings = []
            name_with_end = name + self.separation_token
            for char in name_with_end:
                name_encodings.append(self.stoi[char])
            data.append(name_encodings)

        return data

    def _build_datasets(self) -> None:
        train_data, val_data, test_data = _split_dataset(self.names, self.train_frac, self.val_frac)
        self.train = self._build_single_dataset(train_data)
        self.val = self._build_single_dataset(val_data)
        self.test  = self._build_single_dataset(test_data)


class SentencesDatasetBuilder(DatasetBuilder):
    def __init__(self,
                 dataset_name: str,
                 data_path: str,
                 raw_text: str,
                 separation_token: str,
                 tokenizer: str = 'punkt',
                 sentences_in_fragment: int = 1,
                 max_number_of_tokens_in_fragment: int = 150,
                 train_frac: float = 0.8,
                 val_frac: float = 0.1):
        self.raw_text = raw_text
        self.tokenizer = tokenizer
        self.sentences_in_fragment = sentences_in_fragment
        self.max_number_of_tokens_in_fragment = max_number_of_tokens_in_fragment
        super().__init__(
            dataset_name=dataset_name,
            data_path=data_path,
            separation_token=separation_token,
            train_frac=train_frac,
            val_frac=val_frac
        )

    def _build_alphabet(self) -> None:
        self.alphabet = [self.separation_token] + sorted(set(self.raw_text))

    def _build_datasets(self) -> None:
        fragments = self.parse_text_into_fragments(self.raw_text)
        train_data, val_data, test_data = _split_dataset(fragments, self.train_frac, self.val_frac)
        self.train = self._build_single_dataset(train_data)
        self.val = self._build_single_dataset(val_data)
        self.test = self._build_single_dataset(test_data)

    def parse_text_into_fragments(self, text: str) -> List[str]:
        _ensure_tokenizer_is_downloaded(self.tokenizer)
        sentences = nltk.sent_tokenize(text)
        fragments = []
        i = 0

        while i < len(sentences):
            fragment = "".join(sentences[i: min(i + self.sentences_in_fragment, len(sentences))])
            if len(fragment) <= self.max_number_of_tokens_in_fragment:
                fragments.append(fragment)
            i += self.sentences_in_fragment

        return fragments

    def _build_single_dataset(self, dataset: List[str]) -> List[List[int]]:
        data = []
        for fragment in dataset:
            fragment_encodings = []
            for char in fragment:
                fragment_encodings.append(self.stoi[char])
            fragment_encodings.append(self.stoi[self.separation_token])
            data.append(fragment_encodings)

        return data
