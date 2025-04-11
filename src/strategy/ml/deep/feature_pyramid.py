import torch
import torch.nn as nn

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels: List[int], out_channel: int):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channel in in_channels:
            inner_block = nn.Conv1d(in_channel, out_channel, 1)
            layer_block = nn.Conv1d(out_channel, out_channel, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
