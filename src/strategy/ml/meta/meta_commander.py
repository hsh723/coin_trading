import torch
import torch.nn as nn
from typing import Dict, List

class MetaCommander(nn.Module):
    def __init__(self, state_dim: int, num_strategies: int = 5):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.strategy_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        )
        
        self.strategy_selector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_strategies),
            nn.Softmax(dim=-1)
        )
