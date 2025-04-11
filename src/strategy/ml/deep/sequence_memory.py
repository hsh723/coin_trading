import torch
import torch.nn as nn

class SequenceMemoryNetwork(nn.Module):
    def __init__(self, input_size: int, memory_size: int):
        super().__init__()
        self.memory_size = memory_size
        
        self.memory_controller = nn.LSTM(
            input_size=input_size,
            hidden_size=memory_size,
            num_layers=2,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=memory_size * 2,
            num_heads=4
        )
