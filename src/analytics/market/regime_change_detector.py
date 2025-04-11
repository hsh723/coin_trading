import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MarketRegime:
    current_regime: str  # trending, ranging, volatile
    regime_probability: float
    regime_duration: int
    next_regime_prediction: Dict[str, float]

class RegimeChangeDetector:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.regime_history = []
        
    async def detect_regime(self, market_data: Dict) -> MarketRegime:
        """시장 레짐 변화 감지"""
        return MarketRegime(
            current_regime=self._identify_current_regime(market_data),
            regime_probability=self._calculate_regime_probability(market_data),
            regime_duration=self._calculate_regime_duration(),
            next_regime_prediction=self._predict_next_regime(market_data)
        )
