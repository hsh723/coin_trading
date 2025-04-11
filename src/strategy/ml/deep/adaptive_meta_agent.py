import torch
import torch.nn as nn

class AdaptiveMetaAgent(nn.Module):
    def __init__(self, state_dim: int, task_dim: int):
        super().__init__()
        
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        self.adaptation_network = nn.GRU(
            input_size=160,  # 128 + 32
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
