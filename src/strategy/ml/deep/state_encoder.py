import torch
import torch.nn as nn
import numpy as np

class MarketStateEncoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int):
        super().__init__()
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.LayerNorm(encoding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(encoding_dim * 2, encoding_dim)
        )
        
        self.state_predictor = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU(),
            nn.Linear(encoding_dim // 2, 3)  # 3 possible market states
        )
