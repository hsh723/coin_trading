import numpy as np
import pandas as pd
from typing import Tuple, Dict
from ..analysis.cointegration_analyzer import CointegrationAnalyzer

class PairsTrading:
    def __init__(self, params: Dict):
        self.z_threshold = params.get('z_threshold', 2.0)
        self.cointegration_analyzer = CointegrationAnalyzer()
        
    def find_trading_signals(self, pair1: pd.Series, pair2: pd.Series) -> Dict[str, str]:
        """페어 트레이딩 신호 생성"""
        spread = self._calculate_spread(pair1, pair2)
        z_score = (spread - spread.mean()) / spread.std()
        
        if z_score[-1] > self.z_threshold:
            return {'pair1': 'SELL', 'pair2': 'BUY'}
        elif z_score[-1] < -self.z_threshold:
            return {'pair1': 'BUY', 'pair2': 'SELL'}
        return {'pair1': 'HOLD', 'pair2': 'HOLD'}
