from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class MarketRegime:
    current_regime: str
    regime_probability: float
    transition_matrix: Dict[str, Dict[str, float]]
    volatility_state: str
    trend_state: str

class MarketRegimeAnalyzer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.regimes = ['trend', 'mean_reverting', 'volatile', 'stable']
        
    async def analyze_regime(self, market_data: Dict) -> MarketRegime:
        """시장 레짐 분석"""
        prices = np.array(market_data['prices'])
        returns = np.diff(np.log(prices))
        
        current_regime = self._identify_regime(returns)
        prob = self._calculate_regime_probability(returns, current_regime)
        
        return MarketRegime(
            current_regime=current_regime,
            regime_probability=prob,
            transition_matrix=self._calculate_transition_matrix(returns),
            volatility_state=self._determine_volatility_state(returns),
            trend_state=self._determine_trend_state(prices)
        )
