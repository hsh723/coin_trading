import torch
import torch.nn as nn

class MultiScaleEncoder(nn.Module):
    def __init__(self, input_channels: int, base_channels: int = 64):
        super().__init__()
        
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, base_channels * (2 ** i), 3, padding=1),
                nn.BatchNorm1d(base_channels * (2 ** i)),
                nn.ReLU()
            ) for i in range(3)
        ])
