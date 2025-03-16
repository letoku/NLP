from datasets_utils.names_dataset_builder import NamesDatasetBuilder

DATA_PATH = 'data'

def __main__():
    names = open('names.txt', 'r').read().splitlines()
    builder = NamesDatasetBuilder(names, data_path=DATA_PATH)
    builder.build()


if __name__ == '__main__':
    __main__()
