import torch
import torch.nn as nn
from typing import Dict, Tuple

class RiskAwareNetwork(nn.Module):
    def __init__(self, input_dim: int, risk_config: Dict = None):
        super().__init__()
        
        self.config = risk_config or {
            'risk_weight': 0.3,
            'var_alpha': 0.95
        }
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
        self.risk_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # VaR and CVaR estimates
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Action probabilities
        )
