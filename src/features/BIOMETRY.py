import torchvision
import numpy as np
from ..paths import PATH_DATA_PROCESSED

def loader(filepath):
    return np.load(filepath)

class BIOMETRY(torchvision.datasets.DatasetFolder):

    def __init__(self, num_classes: int, **kwargs):
        kwargs['root'] = PATH_DATA_PROCESSED.joinpath(f'BIOMETRY{num_classes}').as_posix()
        kwargs['loader'] = loader
        kwargs['extensions'] = ('npy',)
        super().__init__(**kwargs)
