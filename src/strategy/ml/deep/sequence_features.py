import torch
import torch.nn as nn

class SequenceFeatureExtractor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
