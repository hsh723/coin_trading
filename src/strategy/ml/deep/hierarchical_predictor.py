import torch
import torch.nn as nn
from typing import Dict, List

class HierarchicalPredictor(nn.Module):
    def __init__(self, input_size: int, num_levels: int = 3):
        super().__init__()
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, input_size * 2),
                nn.LayerNorm(input_size * 2),
                nn.ReLU(),
                nn.Linear(input_size * 2, input_size)
            ) for _ in range(num_levels)
        ])
        
        self.attention = nn.MultiheadAttention(input_size, 4)
        self.final_predictor = nn.Linear(input_size, 1)
