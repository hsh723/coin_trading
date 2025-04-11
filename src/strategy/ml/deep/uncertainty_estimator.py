import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyEstimator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # mean and variance
        )
