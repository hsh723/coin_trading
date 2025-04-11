import torch
import torch.nn as nn

class MarketDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.decoder_rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim * 2,
            num_layers=2,
            batch_first=True
        )
        self.projection = nn.Linear(latent_dim * 2, output_dim)
