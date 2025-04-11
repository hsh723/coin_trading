import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads
        )
        self.feature_projection = nn.Linear(feature_dim, feature_dim)
