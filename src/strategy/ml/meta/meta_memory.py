import torch
import torch.nn as nn
from typing import Dict, List

class EpisodicMemory(nn.Module):
    def __init__(self, memory_size: int, feature_dim: int):
        super().__init__()
        
        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.randn(memory_size, feature_dim))
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=4)
        
        self.memory_updater = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True
        )
