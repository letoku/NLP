from typing import List, Tuple
import random
import nltk


def _assert_correct_fractions(train_frac: float, val_frac: float):
    assert train_frac + val_frac < 1.0, "Train and val fractions sums to >= 1"
    assert train_frac > 0, "Train fraction isn't positive"
    assert val_frac > 0, "Val fraction isn't positive"


def _split_dataset(texts: List[str], train_frac: float = 0.8, val_frac: float = 0.1) ->\
        Tuple[List[str], List[str], List[str]]:

    _assert_correct_fractions(train_frac, val_frac)

    texts_shuffled = random.sample(texts, len(texts))

    train_size = int(len(texts_shuffled) * train_frac)
    val_size = int(len(texts_shuffled) * val_frac)

    train_data = texts_shuffled[:train_size]
    val_data = texts_shuffled[train_size:train_size + val_size]
    test_data = texts_shuffled[train_size + val_size:]

    return train_data, val_data, test_data


def _ensure_tokenizer_is_downloaded(tokenizer: str) -> None:
    try:
        nltk.data.find(f'tokenizers/{tokenizer}')
    except LookupError:
        nltk.download(tokenizer, quiet=True)
