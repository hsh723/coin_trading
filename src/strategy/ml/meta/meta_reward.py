import torch
import torch.nn as nn

class MetaRewardEstimator(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        
        self.reward_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
