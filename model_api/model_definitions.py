import torch

class RNNModel(torch.nn.Module):
    def __init__(self, input_size):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size=32, batch_first=True)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        h0 = torch.zeros(1, x.size(0), 32)  # Initial hidden state
        out, _ = self.rnn(x, h0)
        out = torch.sigmoid(self.fc(out[:, -1, :]))
        return out



