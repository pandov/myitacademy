import torch

from ..paths import PATH_MODELS

class RestModule(object):

    def __init__(self):
        self.summary()

    def summary(self):
        length = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return print(self, f'({length})')

    def init_weights(self):
        checkpoint = torch.load(PATH_MODELS.joinpath(self.pth))
        self.load_state_dict(checkpoint['model_state_dict'])
