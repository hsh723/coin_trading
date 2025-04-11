import torch
import torch.nn as nn

class AdaptivePortfolioManager(nn.Module):
    def __init__(self, num_assets: int, feature_dim: int):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        self.allocation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_assets),
            nn.Softmax(dim=-1)
        )
