import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=state_dim,
                nhead=8,
                dim_feedforward=state_dim * 4
            ),
            num_layers=4
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
