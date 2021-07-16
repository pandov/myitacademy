import torch
from torch import nn
from src.utils import IMG_SIZE

class Conv2dScaling(object):

    def __init__(self, shape):
        if type(shape) is int:
            shape = (shape, shape)
        self.shape = torch.tensor(shape, dtype=torch.int32)

    def __rescale__(self, padding, dilation, kernel_size, stride):
        t = lambda x: torch.tensor(x)
        self.shape = (self.shape + 2 * t(padding) - t(dilation) * (t(kernel_size) - 1) - 1) // t(stride) + 1

    def rescale(self, *layers):
        for layer in layers:
            if layer:
                self.__rescale__(layer.padding, layer.dilation, layer.kernel_size, layer.stride)

    def size(self):
        return torch.prod(self.shape)

class FERCNN(nn.Module):

    def __init__(self, num_classes, channels_1=32, channels_2=64, channels_3=128, channels_4=256):
        super().__init__()
        kwargs = dict(kernel_size=3, stride=1, padding=1)
        conv1 = nn.Conv2d(1, channels_1, **kwargs)
        conv2 = nn.Conv2d(channels_1, channels_2, **kwargs)
        conv3 = nn.Conv2d(channels_2, channels_3, **kwargs)
        conv4 = nn.Conv2d(channels_3, channels_4, **kwargs)
        pool = nn.MaxPool2d(2, 2)

        scaling = Conv2dScaling(IMG_SIZE)
        scaling.rescale(conv1, conv2, pool, conv3, conv4, pool)

        self.conv_bn_relu = nn.Sequential(
            conv1,
            nn.BatchNorm2d(channels_1),
            nn.ReLU(),
            conv2,
            nn.BatchNorm2d(channels_2),
            nn.ReLU(),
            pool,
            conv3,
            nn.BatchNorm2d(channels_3),
            nn.ReLU(),
            conv4,
            nn.BatchNorm2d(channels_4),
            nn.ReLU(),
            pool,
            nn.Dropout2d(0.2),
        )
        self.fc = nn.Linear(scaling.size() * channels_4, num_classes)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
