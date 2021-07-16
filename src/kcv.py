import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Subset, DataLoader
from catalyst.dl import SupervisedRunner

class CrossValidation(object):

    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k

    def __iter__(self):
        length = len(self.dataset)
        valid_size = int(length / self.k)
        train_size = length - valid_size
        idx = np.random.permutation(length).tolist()
        for k in range(self.k):
            start = valid_size * k
            end = start + valid_size
            train_dataset = Subset(self.dataset, idx[:start] + idx[end:])
            valid_dataset = Subset(self.dataset, idx[start:end])
            yield k, train_dataset, valid_dataset

class Experiment(object):

    def model(self):
        raise NotImplementedError()

    def criterion(self):
        raise NotImplementedError()

    def optimizer(self, model):
        raise NotImplementedError()

    def scheduler(self, optimizer):
        raise NotImplementedError()

def experiment_run(experiments, callbacks, cross_val, logdir, device, main_metric, batch_size, num_epochs, num_workers=8):
    for k, train_dataset, valid_dataset in cross_val:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), num_workers=num_workers)
        loaders = {'train': train_loader, 'valid': valid_loader}
        for p, experiment in enumerate(experiments):
            logpath = logdir + f'/p_{p + 1}/k_{k + 1}'
            if Path(logpath).exists(): continue
            runs = {}
            runs['model'] = experiment.model()
            runs['criterion'] = experiment.criterion()
            runs['optimizer'] = experiment.optimizer(runs['model'])
            runs['scheduler'] = experiment.scheduler(runs['optimizer'])
            runs['loaders'] = loaders
            runs['callbacks'] = callbacks
            runs['logdir'] = logpath
            runs['num_epochs'] = num_epochs
            runs['minimize_metric'] = False
            runs['main_metric'] = main_metric
            runner = SupervisedRunner(device=device)
            runner.train(**runs)
