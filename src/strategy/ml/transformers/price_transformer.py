import torch
import torch.nn as nn
from typing import Dict

class PriceTransformer(nn.Module):
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dropout': 0.1
        }
        
        self.transformer = nn.Transformer(
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_encoder_layers=self.config['num_layers'],
            num_decoder_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
