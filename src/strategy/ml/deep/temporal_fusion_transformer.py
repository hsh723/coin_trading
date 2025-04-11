import torch
import torch.nn as nn

class TemporalFusionTransformer(nn.Module):
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {
            'hidden_size': 128,
            'attention_heads': 4,
            'dropout': 0.1,
            'num_layers': 3
        }
        
        self.variable_selection = self._build_variable_selection()
        self.temporal_processing = self._build_temporal_processing()
        self.decoder = self._build_decoder()
