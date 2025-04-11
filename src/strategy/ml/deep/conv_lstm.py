import torch
import torch.nn as nn

class ConvLSTM(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: List[int]):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_channels if i == 0 else hidden_channels[i-1],
                out_channels=hidden_channels[i],
                kernel_size=3,
                padding=1
            ) for i in range(len(hidden_channels))
        ])
        
        self.lstm = nn.LSTM(
            input_size=hidden_channels[-1],
            hidden_size=hidden_channels[-1],
            num_layers=2,
            batch_first=True
        )
