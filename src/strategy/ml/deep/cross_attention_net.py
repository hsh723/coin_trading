import torch
import torch.nn as nn

class CrossAttentionNetwork(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1
        )
