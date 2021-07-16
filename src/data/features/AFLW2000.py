import numpy as np
from scipy.io import loadmat
from src.utils import DIR
from .abc import DatasetWithSubsetLoader
from .transforms import FaceLandmarksSampleNormalize

class AFLW2000(DatasetWithSubsetLoader):
    def __init__(self):
        super().__init__()
        path = DIR.DATA.EXTERNAL.joinpath('AFLW2000')
        self.filepaths = list(path.rglob('*.mat'))
        self.transform = FaceLandmarksSampleNormalize()
        self.indices = self._get_indices()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        mat = loadmat(filepath.as_posix())
        sample = mat['pt3d_68'].T
        sample = self.transform(sample)
        inputs = sample[:, :2]
        targets =  sample[:, 2]
        return inputs, targets
