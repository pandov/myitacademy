import torch

class FMDepthRNN(torch.nn.Module):

    def __init__(self, input_dim=2, hidden_size=128, num_layers=2, output_dim=68, nonlinearity='relu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.rnn = torch.nn.RNN(input_dim, hidden_size, num_layers, nonlinearity=nonlinearity, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_dim)

    def init_hidden(self, x):
        return torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)

    def forward(self, x):
        h0 = self.init_hidden(x)
        x, hn = self.rnn(x, h0)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
