import torch
import numpy as np
from torchvision import transforms
from scipy.io import loadmat
from ..transforms import FaceLandmarksNormalize
from ..paths import PATH_DATA_EXTERNAL

class AFLW2000(torch.utils.data.Dataset):

    def __init__(self):
        path = PATH_DATA_EXTERNAL.joinpath('AFLW2000')
        self.filepaths = sorted(list(path.rglob('*.mat')))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        mat = loadmat(filepath.as_posix())
        sample = mat['pt3d_68'].T
        sample = self.transform([sample])
        inputs = sample[0, :, :2]
        targets =  sample[0, :, 2]
        return inputs, targets

    @property
    def transform(self):
        return transforms.Compose([
            FaceLandmarksNormalize(),
            torch.FloatTensor,
        ])
