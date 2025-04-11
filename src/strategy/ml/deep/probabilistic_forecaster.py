import torch
import torch.nn as nn
from typing import Dict, Tuple

class ProbabilisticForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.mixture_head = nn.Linear(hidden_dim, 3)
