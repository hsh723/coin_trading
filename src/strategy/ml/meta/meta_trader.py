import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class MetaTrader(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.market_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.task_embedding = nn.Parameter(torch.randn(3, hidden_dim))  # 3개의 기본 거래 전략
        
        self.strategy_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(3)
        ])
