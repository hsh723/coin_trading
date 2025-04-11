import torch
import torch.nn as nn
from typing import Dict, List

class TransformerRLModel(nn.Module):
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {
            'input_dim': 128,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'action_dim': 3
        }
        
        self.input_projection = nn.Linear(self.config['input_dim'], 256)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=self.config['num_heads'],
                dim_feedforward=1024,
                dropout=self.config['dropout']
            ),
            num_layers=self.config['num_layers']
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.config['action_dim']),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
