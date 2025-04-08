from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..analysis.technical import TechnicalAnalyzer

class VolatilityStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.tech_analyzer = TechnicalAnalyzer()
        self.vol_window = config.get('volatility_window', 20)
        self.vol_threshold = config.get('volatility_threshold', 2.0)
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """변동성 기반 신호 생성"""
        atr = self.tech_analyzer.calculate_atr(data, self.vol_window)
        current_vol = atr.iloc[-1]
        avg_vol = atr.mean()
        
        if current_vol > avg_vol * self.vol_threshold:
            return self._generate_breakout_signals(data)
        else:
            return self._generate_range_signals(data)
