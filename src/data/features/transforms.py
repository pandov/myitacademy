import torch

class FaceLandmarksSampleNormalize(object):
    def __call__(self, array):
        sample = torch.tensor(array, dtype=torch.float32)

        tmin = torch.min(sample, dim=0).values
        tmax = torch.max(sample, dim=0).values

        center = torch.div(tmin + tmax, 2)
        shape = torch.div(tmax - tmin, 2)

        sample = torch.sub(sample, center)
        sample = torch.div(sample, shape)
        return sample
