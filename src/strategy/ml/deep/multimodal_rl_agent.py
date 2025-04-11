import torch
import torch.nn as nn

class MultiModalRLAgent(nn.Module):
    def __init__(self, state_dims: Dict[str, int]):
        super().__init__()
        
        self.modal_encoders = nn.ModuleDict({
            'price': nn.LSTM(state_dims['price'], 64, batch_first=True),
            'volume': nn.LSTM(state_dims['volume'], 32, batch_first=True),
            'sentiment': nn.LSTM(state_dims['sentiment'], 32, batch_first=True)
        })
        
        self.attention_fusion = MultiModalAttention()
        
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Buy, Sell, Hold
        )
