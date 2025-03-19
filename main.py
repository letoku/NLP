from datetime import datetime
from data_processing.datasets_builders import NamesDatasetBuilder


DATA_PATH = 'data'
SOURCE_FILE = 'raw_data/names.txt'


def __main__():
    now = datetime.now()
    time_of_building = now.strftime("%Y-%m-%d")
    names = open(SOURCE_FILE, 'r').read().splitlines()
    builder = NamesDatasetBuilder(
        dataset_name=f"names_{time_of_building}",
        names=names,
        data_path=DATA_PATH,
        separation_token='.'
    )
    builder.build()


if __name__ == '__main__':
    __main__()
