from names_dataset_builder import NamesDatasetBuilder
from torch.utils.data import DataLoader


def __main__():
    names = open('names.txt', 'r').read().splitlines()
    builder = NamesDatasetBuilder(names, block_size=3)
    builder.build()


if __name__ == '__main__':
    __main__()
