import torch
import torch.nn.functional as F

from .rest import RestModule

class Conv2dScaling(object):

    def __init__(self, shape):
        if type(shape) is int:
            shape = (shape, shape)
        self.shape = torch.tensor(shape, dtype=torch.int32)

    def __rescale__(self, padding, dilation, kernel_size, stride):
        t = lambda x: torch.tensor(x)
        self.shape = (self.shape + 2 * t(padding) - t(dilation) * (t(kernel_size) - 1) - 1) / t(stride) + 1

    def rescale(self, *layers):
        for layer in layers:
            if layer:
                self.__rescale__(layer.padding, layer.dilation, layer.kernel_size, layer.stride)

    def size(self):
        return torch.prod(self.shape)

class FERCNN(torch.nn.Module, RestModule):

    def __init__(self, num_classes, channels_1=12, channels_2=24, channels_3=48, channels_4=72): # 6
        super().__init__()
        self.pth = f'fercnn{num_classes}.pth'
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(1, channels_1, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels_1)
        self.dropout1 = torch.nn.Dropout2d(0.2)
        self.conv2 = torch.nn.Conv2d(channels_1, channels_2, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels_2)
        self.dropout2 = torch.nn.Dropout2d(0.4)
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = torch.nn.Conv2d(channels_2, channels_3, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(channels_3)
        self.dropout3 = torch.nn.Dropout2d(0.2)
        self.conv4 = torch.nn.Conv2d(channels_3, channels_4, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(channels_4)
        self.dropout4 = torch.nn.Dropout2d(0.4)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        scaling = Conv2dScaling(48)
        scaling.rescale(self.conv1, self.conv2, self.pool1, self.conv3, self.conv4, self.pool2)

        self.fc = torch.nn.Linear(scaling.size() * channels_4, num_classes)

    def __len__(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.dropout4(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
