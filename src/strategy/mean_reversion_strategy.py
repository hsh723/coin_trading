from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..analysis.technical import TechnicalAnalyzer

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, params: Dict):
        super().__init__()
        self.tech_analyzer = TechnicalAnalyzer()
        self.lookback_period = params.get('lookback_period', 20)
        self.std_dev_threshold = params.get('std_dev_threshold', 2.0)
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """평균회귀 기반 거래 신호 생성"""
        sma = data['close'].rolling(window=self.lookback_period).mean()
        std_dev = data['close'].rolling(window=self.lookback_period).std()
        
        upper_band = sma + (std_dev * self.std_dev_threshold)
        lower_band = sma - (std_dev * self.std_dev_threshold)
        
        current_price = data['close'].iloc[-1]
        
        if current_price > upper_band.iloc[-1]:
            return {'action': 'SELL'}
        elif current_price < lower_band.iloc[-1]:
            return {'action': 'BUY'}
        return {'action': 'HOLD'}
