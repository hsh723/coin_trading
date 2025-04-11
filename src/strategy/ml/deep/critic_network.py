import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        self.q_network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
