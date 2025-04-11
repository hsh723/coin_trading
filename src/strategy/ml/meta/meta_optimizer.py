import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaOptimizer(nn.Module):
    def __init__(self, model_params_dim: int):
        super().__init__()
        
        self.lstm_optimizer = nn.LSTM(
            input_size=model_params_dim * 2,  # 파라미터와 그래디언트
            hidden_size=model_params_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.update_network = nn.Sequential(
            nn.Linear(model_params_dim, model_params_dim // 2),
            nn.ReLU(),
            nn.Linear(model_params_dim // 2, model_params_dim)
        )
