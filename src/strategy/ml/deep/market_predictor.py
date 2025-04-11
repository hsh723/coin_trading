import torch
import torch.nn as nn

class MarketStatePredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.feature_extractor = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=True
        )
        
        self.predictor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Up, Down, Sideways
        )
