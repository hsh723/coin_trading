import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.pooling = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
