import torch
import torch.nn as nn
import numpy as np

class RiskAwareAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        self.risk_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        
        self.action_generator = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        self.risk_assessor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
