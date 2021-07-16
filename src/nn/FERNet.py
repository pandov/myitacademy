import torch

from .FERCNN import FERCNN
from .FERFC import FERFC

class FERNet(torch.nn.Module):

    def __init__(self, num_classes=7, hidden_size=1024):
        super().__init__()

        fc = FERFC()
        fc.requires_grad_(False)
        fc.fc = torch.nn.Linear(fc.fc.in_features, hidden_size)
        self.fc = fc

        cnn = FERCNN()
        cnn.requires_grad_(False)
        cnn.fc = torch.nn.Linear(cnn.fc.in_features, hidden_size)
        self.cnn = cnn

        self.out = torch.nn.Sequential(
            torch.nn.BatchNorm1d(hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, t):
        x, y = t
        x = self.cnn(x)
        y = self.fc(y)
        z = torch.cat((x, y), dim=1)
        z = self.out(z)
        return z
