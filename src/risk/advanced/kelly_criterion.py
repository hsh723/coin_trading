import numpy as np
from typing import Tuple

class KellyCriterion:
    def __init__(self, max_position_size: float = 0.5):
        self.max_position_size = max_position_size

    def calculate_position_size(self, win_rate: float, win_loss_ratio: float) -> float:
        """켈리 기준 포지션 크기 계산"""
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        return min(max(kelly_fraction, 0), self.max_position_size)

    def calculate_optimal_fraction(self, returns: np.ndarray) -> Tuple[float, dict]:
        """최적 배분 비율 계산"""
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns)
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        kelly_fraction = self.calculate_position_size(win_rate, win_loss_ratio)
        
        return kelly_fraction, {
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
