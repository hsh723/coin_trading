from typing import Dict, List
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class MarketRegimeSignal:
    current_regime: str
    regime_probability: float
    transition_matrix: Dict[str, Dict[str, float]]
    volatility_state: str

class MarketRegimeDetector:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'window_size': 60,
            'regime_types': ['trending', 'mean_reverting', 'volatile'],
            'min_regime_duration': 5
        }
        
    async def detect_regime(self, market_data: pd.DataFrame) -> MarketRegimeSignal:
        """시장 국면 감지"""
        returns = self._calculate_returns(market_data['close'])
        volatility = self._calculate_volatility(returns)
        
        return MarketRegimeSignal(
            current_regime=self._identify_regime(returns, volatility),
            regime_probability=self._calculate_regime_probability(returns),
            transition_matrix=self._calculate_transition_matrix(),
            volatility_state=self._determine_volatility_state(volatility)
        )
