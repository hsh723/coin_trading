import torch
import torch.nn as nn

class HybridForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.predictor = nn.Linear(hidden_size, 1)
