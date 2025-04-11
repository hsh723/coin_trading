import torch
import torch.nn as nn
from typing import Dict, Tuple

class Generator(nn.Module):
    def __init__(self, latent_dim: int, seq_length: int):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, seq_length * 64)
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1
        )
        self.decoder = nn.GRU(64, 32, num_layers=2, batch_first=True)
        self.final_proj = nn.Linear(32, 1)

class Discriminator(nn.Module):
    def __init__(self, seq_length: int):
        super().__init__()
        self.feature_extractor = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8
        )
