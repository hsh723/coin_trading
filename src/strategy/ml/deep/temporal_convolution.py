import torch
import torch.nn as nn
from typing import List, Dict

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, channels: List[int], kernel_size: int = 2):
        super().__init__()
        layers = []
        num_levels = len(channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else channels[i-1]
            out_channels = channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, 
                    kernel_size, dilation=dilation
                )
            )
            
        self.network = nn.Sequential(*layers)
