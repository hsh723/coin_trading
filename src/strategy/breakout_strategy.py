from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..analysis.technical import TechnicalAnalyzer

class BreakoutStrategy(BaseStrategy):
    def __init__(self, params: Dict):
        super().__init__()
        self.tech_analyzer = TechnicalAnalyzer()
        self.lookback_period = params.get('lookback_period', 20)
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """브레이크아웃 신호 생성"""
        high = data['high'].rolling(window=self.lookback_period).max()
        low = data['low'].rolling(window=self.lookback_period).min()
        
        signals = {}
        current_price = data['close'].iloc[-1]
        
        if current_price > high.iloc[-2]:  # 상향 브레이크아웃
            signals['action'] = 'BUY'
        elif current_price < low.iloc[-2]:  # 하향 브레이크아웃
            signals['action'] = 'SELL'
        else:
            signals['action'] = 'HOLD'
            
        return signals
