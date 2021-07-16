import os
import torch
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader

class DatasetWithSubsetLoader(Dataset):
    def data_loader(self, start=None, end=None, **kwargs):
        dataset = Subset(self, self.indices[start:end])
        dataloader = DataLoader(dataset, **kwargs)
        return dataloader

    def _get_indices(self):
        return np.random.permutation(len(self))

def _find_classes(path, exclude):
    classes = [d.name for d in os.scandir(path) if d.is_dir() and d.name not in exclude]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def _weight_classes(path, classes):
    count = [len(os.listdir(os.path.join(path, cls_name))) for cls_name in classes]
    count = torch.tensor(count, dtype=torch.float32)
    probs = count / count.sum()
    inv = 1 / probs
    probs = inv / inv.sum()
    res = probs / probs.min()
    return res
