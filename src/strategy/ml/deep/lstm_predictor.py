import torch
import torch.nn as nn
from typing import Dict, List

class LSTMPredictor(nn.Module):
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {
            'input_size': 10,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2
        }
        
        self.lstm = nn.LSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            batch_first=True
        )
        self.fc = nn.Linear(self.config['hidden_size'], 1)
