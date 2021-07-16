from types import SimpleNamespace
from pathlib import Path

__mainpath__ = Path(__file__).absolute().parent.parent.parent
__datapath__ = __mainpath__.joinpath('data')
DIR = SimpleNamespace(
    DATA = SimpleNamespace(
        EXTERNAL = __datapath__.joinpath('external'),
        PROCESSED = __datapath__.joinpath('processed'),
    ),
    MODELS = __datapath__.joinpath('models'),
)
