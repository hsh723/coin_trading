from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

class DQNModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """예측 실행"""
        return self.network(state)
