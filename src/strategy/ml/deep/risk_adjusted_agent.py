import torch
import torch.nn as nn
import numpy as np

class RiskAdjustedAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        self.risk_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
