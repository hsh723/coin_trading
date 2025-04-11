import torch
import torch.nn as nn
from typing import Dict, Tuple

class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.predictor = nn.Linear(hidden_dim, 1)
        
    async def forecast(self, 
                      sequence: torch.Tensor, 
                      future_steps: int = 10) -> Tuple[torch.Tensor, float]:
        """시계열 예측 실행"""
        predictions = []
        confidence_scores = []
        
        with torch.no_grad():
            hidden = None
            for _ in range(future_steps):
                output, hidden = self.gru(sequence, hidden)
                pred = self.predictor(output[:, -1, :])
                predictions.append(pred)
                
        return torch.stack(predictions, dim=1), self._calculate_confidence(predictions)
