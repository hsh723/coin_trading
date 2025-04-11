import torch
import torch.nn as nn

class MarketContextEncoder(nn.Module):
    def __init__(self, input_dim: int, context_dim: int):
        super().__init__()
        
        self.price_encoder = nn.Sequential(
            nn.Linear(input_dim, context_dim * 2),
            nn.LayerNorm(context_dim * 2),
            nn.GELU(),
            nn.Linear(context_dim * 2, context_dim)
        )
        
        self.volume_encoder = nn.Sequential(
            nn.Linear(input_dim, context_dim * 2),
            nn.LayerNorm(context_dim * 2),
            nn.GELU(),
            nn.Linear(context_dim * 2, context_dim)
        )
