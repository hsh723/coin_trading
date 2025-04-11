import torch
import torch.nn as nn
from typing import Dict, List

class MarketContextEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=0.1
        )
