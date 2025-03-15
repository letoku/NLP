import names_dataset


def __main__():
    names = open('names.txt', 'r').read().splitlines()
    builder = names_dataset.NamesDatasetBuilder(names, block_size=2)
    builder.build()


if __name__ == '__main__':
    __main__()
