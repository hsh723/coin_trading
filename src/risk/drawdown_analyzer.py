import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DrawdownMetrics:
    current_drawdown: float
    max_drawdown: float
    recovery_time: int
    drawdown_periods: List[Dict]

class DrawdownAnalyzer:
    def __init__(self, recovery_threshold: float = 0.0):
        self.recovery_threshold = recovery_threshold
        
    def analyze_drawdowns(self, equity_curve: pd.Series) -> DrawdownMetrics:
        """손실폭 분석"""
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        
        drawdown_periods = self._find_drawdown_periods(drawdowns)
        current_drawdown = drawdowns.iloc[-1]
        
        return DrawdownMetrics(
            current_drawdown=current_drawdown,
            max_drawdown=drawdowns.min(),
            recovery_time=self._calculate_recovery_time(drawdowns),
            drawdown_periods=drawdown_periods
        )
