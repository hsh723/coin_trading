import pandas as pd
from typing import Dict
from dataclasses import dataclass

@dataclass
class IchimokuSignals:
    trend_strength: float
    cloud_status: str
    support_levels: list
    resistance_levels: list
    signals: Dict[str, str]

class IchimokuAnalyzer:
    def __init__(self):
        self.params = {
            'tenkan': 9,
            'kijun': 26,
            'senkou_span_b': 52
        }
        
    def analyze_ichimoku(self, market_data: pd.DataFrame) -> IchimokuSignals:
        """일목균형표 분석"""
        highs = market_data['high']
        lows = market_data['low']
        
        tenkan = self._calculate_tenkan(highs, lows)
        kijun = self._calculate_kijun(highs, lows)
        senkou_span_a = self._calculate_senkou_span_a(tenkan, kijun)
        senkou_span_b = self._calculate_senkou_span_b(highs, lows)
        
        return self._generate_signals(
            market_data['close'].iloc[-1],
            tenkan,
            kijun,
            senkou_span_a,
            senkou_span_b
        )
