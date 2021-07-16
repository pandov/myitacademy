import torch

from .rest import RestModule

class FMDepthRNN(torch.nn.Module, RestModule):

    def __init__(self, input_dim=2, hidden_size=136, num_layers=1, output_dim=68, nonlinearity='tanh'):
        super().__init__()
        self.pth = 'fmdepthrnn.pth'
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.rnn = torch.nn.RNN(input_dim, hidden_size, num_layers, nonlinearity=nonlinearity, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_dim)

    def init_hidden(self, x):
        return torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
