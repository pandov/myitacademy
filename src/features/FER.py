import torchvision
from ..paths import PATH_DATA_PROCESSED

class FER(torchvision.datasets.ImageFolder):

    def __init__(self, num_classes: int, **kwargs):
        kwargs['root'] = PATH_DATA_PROCESSED.joinpath(f'FER{num_classes}').as_posix()
        super().__init__(**kwargs)
