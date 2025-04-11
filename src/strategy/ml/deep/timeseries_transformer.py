import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {
            'input_size': 32,
            'hidden_size': 128,
            'num_layers': 3,
            'num_heads': 8
        }
        
        self.time_embedding = nn.Linear(1, self.config['input_size'])
        self.value_embedding = nn.Linear(1, self.config['input_size'])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config['input_size'],
                nhead=self.config['num_heads']
            ),
            num_layers=self.config['num_layers']
        )
