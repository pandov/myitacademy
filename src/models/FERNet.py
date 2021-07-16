import torch

from .rest import RestModule
from .FERCNN import FERCNN
from .FERFC import FERFC

class FERNet(torch.nn.Module, RestModule):

    def __init__(self, num_classes, hidden_size=512):
        super().__init__()
        self.pth = f'fernet{num_classes}.pth'
        self.num_classes = num_classes

        fc = FERFC(num_classes=num_classes)
        fc.init_weights()
        fc.requires_grad_(False)
        fc.fc2 = torch.nn.Linear(fc.fc2.in_features, hidden_size)
        self.fc = fc

        cnn = FERCNN(num_classes=num_classes)
        cnn.init_weights()
        cnn.requires_grad_(False)
        cnn.fc = torch.nn.Linear(cnn.fc.in_features, hidden_size)
        self.cnn = cnn

        self.out = torch.nn.Linear(hidden_size * 2, num_classes)

    def forward(self, t):
        x, y = t

        x = self.cnn(x)
        y = self.fc(y)

        z = torch.cat((x, y), dim=1)
        z = torch.tanh(z)
        z = self.out(z)

        return z
