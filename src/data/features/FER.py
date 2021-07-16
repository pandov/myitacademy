from torchvision.datasets import ImageFolder
from src.utils import DIR
from .abc import DatasetWithSubsetLoader, _find_classes, _weight_classes

class FER(ImageFolder, DatasetWithSubsetLoader):

    def __init__(self, exclude=None, **kwargs):
        kwargs['root'] = DIR.DATA.PROCESSED.joinpath('FER').as_posix()
        if exclude is None:
            exclude = []
        self.exclude = exclude
        super().__init__(**kwargs)
        self.indices = self._get_indices()

    def _find_classes(self, path):
        return _find_classes(path, self.exclude)

    def _weight_classes(self):
        return _weight_classes(self.root, self.classes)
