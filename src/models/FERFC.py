import torch
import torch.nn.functional as F

from .rest import RestModule

class FERFC(torch.nn.Module, RestModule):

    def __init__(self, num_classes, input_size=163, hidden_size=284):
        super().__init__()
        self.pth = f'ferfc{num_classes}.pth'
        self.num_classes = num_classes
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
