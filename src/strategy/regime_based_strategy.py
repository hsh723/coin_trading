from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..analysis.market_regime import MarketRegimeAnalyzer

class RegimeBasedStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.strategies = {
            'LOW_VOL': self._generate_mean_reversion_signals,
            'HIGH_VOL': self._generate_trend_following_signals,
            'CRISIS': self._generate_defensive_signals
        }
        
    async def generate_signals(self, market_data: pd.DataFrame) -> Dict:
        """시장 국면에 따른 신호 생성"""
        current_regime = self.regime_analyzer.identify_regime(market_data)
        strategy = self.strategies.get(current_regime, self._generate_neutral_signals)
        
        signals = await strategy(market_data)
        signals['regime'] = current_regime
        
        return signals
