import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class MetaReinforcementLearner(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        self.task_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.policy_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            ) for _ in range(3)  # 3개의 서브 정책
        ])
