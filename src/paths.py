from pathlib import Path

PATH_MAIN = Path(__file__).absolute().parent.parent
PATH_MODELS = PATH_MAIN.joinpath('models')
PATH_DATA = PATH_MAIN.joinpath('data')
PATH_DATA_EXTERNAL = PATH_DATA.joinpath('external')
PATH_DATA_PROCESSED = PATH_DATA.joinpath('processed')

def iter_files(path: Path, include: callable = lambda x: True):
    for x in path.iterdir():
        if x.is_file() and include(x):
            yield x
        else:
            yield from iter_files(path / x, include)
