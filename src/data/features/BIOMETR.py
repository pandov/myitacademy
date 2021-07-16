import numpy as np
from torchvision.datasets import DatasetFolder
from src.utils import DIR
from .abc import DatasetWithSubsetLoader, _find_classes, _weight_classes

def loader(filepath):
    return np.load(filepath)

class BIOMETR(DatasetFolder, DatasetWithSubsetLoader):
    def __init__(self, exclude=None, **kwargs):
        kwargs['root'] = DIR.DATA.PROCESSED.joinpath('BIOMETR').as_posix()
        kwargs['loader'] = loader
        kwargs['extensions'] = ('npy',)
        if exclude is None:
            exclude = []
        self.exclude = exclude
        super().__init__(**kwargs)
        self.indices = self._get_indices()

    def _find_classes(self, path):
        return _find_classes(path, self.exclude)

    def _weight_classes(self):
        return _weight_classes(self.root, self.classes)
