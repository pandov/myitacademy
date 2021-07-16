import torch
import torchvision
import numpy as np
from torchvision.datasets.folder import default_loader
from ..utils import *
from ..paths import PATH_DATA_PROCESSED

def loader(filepath):
    image = default_loader(filepath)
    biometry = np.load(filepath.replace('png', 'npy'))
    biometry = torch.FloatTensor(biometry)
    return image, biometry

class CASCADE(torchvision.datasets.ImageFolder):

    def __init__(self, num_classes: int, **kwargs):
        kwargs['root'] = PATH_DATA_PROCESSED.joinpath(f'CASCADE{num_classes}').as_posix()
        kwargs['loader'] = loader
        super().__init__(**kwargs)

    def __getitem__(self, index):
        filepath, targets = self.samples[index]
        image, biometry = self.loader(filepath)
        if self.transform is not None:
            image = self.transform(image)
        return (image, biometry), targets
