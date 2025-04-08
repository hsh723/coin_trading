import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class DrawdownStats:
    current_drawdown: float
    max_drawdown: float
    drawdown_duration: int
    recovery_threshold: float

class DrawdownManager:
    def __init__(self, max_drawdown_limit: float = 0.2):
        self.max_drawdown_limit = max_drawdown_limit
        self.peak = 0.0
        
    def update(self, current_value: float) -> DrawdownStats:
        """손실폭 업데이트 및 분석"""
        self.peak = max(self.peak, current_value)
        current_drawdown = (self.peak - current_value) / self.peak
        
        return DrawdownStats(
            current_drawdown=current_drawdown,
            max_drawdown=max(self.max_drawdown_limit, current_drawdown),
            drawdown_duration=self._calculate_duration(),
            recovery_threshold=self.peak
        )
