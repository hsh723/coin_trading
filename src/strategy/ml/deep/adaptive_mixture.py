import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMixtureModel(nn.Module):
    def __init__(self, feature_dim: int, num_components: int = 3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        self.mixture_params = nn.ModuleList([
            nn.Linear(feature_dim, 2) for _ in range(num_components)
        ])
