from typing import List
import os
from datetime import datetime
from data_processing.datasets_builders import NamesDatasetBuilder, SentencesDatasetBuilder


DATA_PATH = 'data'
SOURCE_DIR = 'raw_data'
NAMES_DATASET_SOURCE_FILE = 'raw_data/names.txt'
OPTIONS = ['names_dataset', 'books_dataset']
BOOKS = ["lalka-tom-pierwszy.txt", "lalka-tom-drugi.txt"]


def build_names_dataset() -> None:
    now = datetime.now()
    time_of_building = now.strftime("%Y-%m-%d")
    names = open(NAMES_DATASET_SOURCE_FILE, 'r').read().splitlines()
    builder = NamesDatasetBuilder(
        dataset_name=f"names__{time_of_building}",
        names=names,
        data_path=DATA_PATH,
        separation_token='.'
    )
    builder.build()
    print("Names dataset built!")


def build_books_dataset(books: List[str]) -> None:
    now = datetime.now()
    time_of_building = now.strftime("%Y-%m-%d")
    text = ''
    for book in books:
        book_text = open(os.path.join(SOURCE_DIR, book), 'r').read()
        text += book_text
        text += '\n'

    books_names = [book.split(".")[0] for book in books]
    builder = SentencesDatasetBuilder(
        dataset_name=f"books__{"&".join(books_names)}__{time_of_building}",
        data_path=DATA_PATH,
        raw_text=text,
        separation_token="<END>",
        tokenizer = 'punkt',
        sentences_in_fragment = 1,
        train_frac = 0.8,
        val_frac = 0.1
    )

    builder.build()
    print("Books dataset built!")


def __main__():
    while True:
        print("\n=== Choose dataset to create ===")
        for i, opt in enumerate(OPTIONS):
            print(f"{i + 1}: {opt}")

        choice = input(f"Enter your choice (1-{len(OPTIONS)}): ").strip()

        if choice == "1":
            build_names_dataset()
            break
        elif choice == "2":
            build_books_dataset(BOOKS)
            break
        else:
            print(f"Invalid choice! Please enter a number between 1 and {len(OPTIONS)}.")


if __name__ == '__main__':
    __main__()
