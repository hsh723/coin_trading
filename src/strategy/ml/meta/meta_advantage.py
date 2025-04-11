import torch
import torch.nn as nn
from typing import Dict, Tuple

class AdvantageEstimator(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, 3)  # 행동 공간 크기
        
    async def estimate(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """상태 가치와 어드밴티지 추정"""
        features = self.feature_network(state)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        return value, advantage
