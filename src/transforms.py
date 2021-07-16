import torch
import numpy as np

class FaceLandmarksNormalize(object):
    def __call__(self, sample):
        sample = np.asarray(sample, dtype=np.float32)

        _min = sample.min(axis=1)
        _max = sample.max(axis=1)

        center = (_min + _max) / 2
        center = np.expand_dims(center, axis=1)
        if center.shape[2] > 2:
            center[:, :, 2] = 0

        shape = (_max - _min) / 2
        if shape.shape[1] > 2:
            shape[:, 2] = 1

        scale = np.prod(shape, axis=1)
        scale = np.sqrt(scale)
        scale = scale[:, np.newaxis, np.newaxis]

        sample -= center
        sample /= scale
        return sample
