import torch
import torchvision
import numpy as np
from torchvision.datasets import DatasetFolder, folder
from src.utils import DIR
from os import scandir
from .abc import DatasetWithSubsetLoader, _find_classes, _weight_classes

def loader(filepath):
    imagepath = filepath.replace('npy', 'png')
    image = folder.default_loader(image)
    biometr = np.load(filepath)
    biometr = torch.from_numpy(biometr).float()
    return image, biometr

class CASCADE(DatasetFolder, DatasetWithSubsetLoader):

    def __init__(self, exclude=None, **kwargs):
        kwargs['root'] = DIR.DATA.PROCESSED.joinpath('CASCADE').as_posix()
        kwargs['loader'] = loader
        kwargs['extensions'] = ('npy',)
        if exclude is None:
            exclude = []
        self.exclude = exclude
        super().__init__(**kwargs)
        self.indices = self._get_indices()

    def __getitem__(self, index):
        path, target = self.samples[index]
        image, biometr = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return (image, biometr), target

    def _find_classes(self, path):
        return _find_classes(path, self.exclude)

    def _weight_classes(self):
        return _weight_classes(self.root, self.classes)
