import torch
import torch.nn as nn

class WaveNetModel(nn.Module):
    def __init__(self, input_channels: int, residual_channels: int, skip_channels: int):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(residual_channels, residual_channels, 2, dilation=2**i)
            for i in range(8)
        ])
        
        self.input_conv = nn.Conv1d(input_channels, residual_channels, 1)
        self.output_conv = nn.Conv1d(skip_channels, 1, 1)
