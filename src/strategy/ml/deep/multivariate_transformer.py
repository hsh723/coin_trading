import torch
import torch.nn as nn
from typing import Dict, List

class MultivariateTransformer(nn.Module):
    def __init__(self, num_features: int, seq_length: int):
        super().__init__()
        
        self.feature_projection = nn.Linear(num_features, 128)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, 128))
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1
            ),
            num_layers=4
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_features)
        )
