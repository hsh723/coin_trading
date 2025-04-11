import torch
import torch.nn as nn

class MetaEnsemble(nn.Module):
    def __init__(self, num_models: int, feature_dim: int):
        super().__init__()
        
        self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=0.1
        )
        
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_models),
            nn.Softmax(dim=-1)
        )
