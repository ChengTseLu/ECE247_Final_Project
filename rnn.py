import torch.nn as nn

DROPOUT=0.5

class LSTMC(nn.Module):
    def __init__(self, input_size=22, hidden_size=1024, num_layers=2, num_classes=4):
        super(LSTMC, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=DROPOUT)

        self.linear = nn.Linear(hidden_size, 4)
    def forward(self, x):

        # Forward propagate RNN
        output, _ = self.lstm(x.permute(0,2,1))
        # Decode hidden state of last time step
        out = self.linear(output[:, -1, :])
        return out


class GRUC(nn.Module):
    def __init__(self, input_size=22, hidden_size=16, num_layers=1, num_classes=4):
        super(GRUC, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=DROPOUT)

        self.linear = nn.Linear(hidden_size, 4)
    def forward(self, x):

        # Forward propagate RNN
        output, _ = self.lstm(x.permute(0,2,1))
        # Decode hidden state of last time step
        out = self.linear(output[:, -1, :])
        return out