import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(initial, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        value = self.stack(x)
        return value
    
class RNN(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.recurrent1 = nn.RNN(input_size=initial, hidden_size=128, num_layers=1)
        self.relu1 = nn.ReLU()
        self.recurrent2 = nn.RNN(input_size=128, hidden_size=64, num_layers=1)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x, hidden1 = self.recurrent1(x)
        x = self.relu1(x)
        x, hidden2 = self.recurrent2(x)
        x = self.relu2(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x
        
class GRU(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.gru1 = nn.GRU(input_size=initial, hidden_size=128, num_layers=1)
        self.relu1 = nn.ReLU()
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, num_layers=1)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x, hidden1 = self.gru1(x)
        x = self.relu1(x)
        x, hidden2 = self.gru2(x)
        x = self.relu2(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x

class LSTM(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=initial, hidden_size=128, num_layers=1)
        self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x, hidden1 = self.lstm1(x)
        x = self.relu1(x)
        x, hidden2 = self.lstm2(x)
        x = self.relu2(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x

class CNN(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x
