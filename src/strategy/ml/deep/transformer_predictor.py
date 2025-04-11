import torch
import torch.nn as nn
from typing import Dict, List

class MarketTransformerPredictor(nn.Module):
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {
            'n_features': 32,
            'n_heads': 8,
            'n_layers': 6,
            'dropout': 0.1
        }
        
        self.feature_embedding = nn.Linear(self.config['n_features'], 512)
        self.position_embedding = nn.Embedding(1000, 512)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=self.config['n_heads'],
                dropout=self.config['dropout']
            ),
            num_layers=self.config['n_layers']
        )
