from torch import nn

class FERFC(nn.Module):

    def __init__(self, num_classes, input_size=163, hidden_size=256):
        super().__init__()
        self.linear_bn_relu = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # nn.LeakyReLU(0.05),
            # nn.Dropout(0.25),
            nn.Tanh(),
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.linear_bn_relu(x)
        x = self.fc(x)
        return x
