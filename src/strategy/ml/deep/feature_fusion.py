import torch
import torch.nn as nn

class FeatureFusionModule(nn.Module):
    def __init__(self, feature_dims: List[int]):
        super().__init__()
        
        self.fusion_layers = nn.ModuleList([
            nn.Linear(dim, dim // 2) for dim in feature_dims
        ])
        
        total_dim = sum(dim // 2 for dim in feature_dims)
        self.final_fusion = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Linear(total_dim // 2, total_dim // 4)
        )
