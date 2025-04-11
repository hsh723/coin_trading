import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

class PortfolioManagerNetwork(nn.Module):
    def __init__(self, num_assets: int, feature_dim: int):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.portfolio_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_assets),
            nn.Softmax(dim=-1)
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
