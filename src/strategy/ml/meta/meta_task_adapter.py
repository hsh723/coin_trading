import torch
import torch.nn as nn
from typing import Dict, List

class MetaTaskAdapter(nn.Module):
    def __init__(self, input_dim: int, num_tasks: int):
        super().__init__()
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.task_specific_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            ) for _ in range(num_tasks)
        ])
        
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1
        )
