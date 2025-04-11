import torch
import torch.nn as nn

class DualAttentionNetwork(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.price_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=0.1
        )
        
        self.volume_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=0.1
        )
        
        self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)
