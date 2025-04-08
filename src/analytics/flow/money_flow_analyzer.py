import pandas as pd
from typing import Dict
from dataclasses import dataclass

@dataclass
class MoneyFlowMetrics:
    mfi_value: float
    net_flow: float
    flow_strength: str
    accumulation: bool

class MoneyFlowAnalyzer:
    def __init__(self, period: int = 14):
        self.period = period
        
    def analyze_money_flow(self, ohlcv_data: pd.DataFrame) -> MoneyFlowMetrics:
        """자금 흐름 지표 분석"""
        typical_price = (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
        money_flow = typical_price * ohlcv_data['volume']
        
        mfi = self._calculate_mfi(money_flow, typical_price)
        net_flow = self._calculate_net_flow(money_flow)
        
        return MoneyFlowMetrics(
            mfi_value=mfi,
            net_flow=net_flow,
            flow_strength=self._determine_strength(mfi),
            accumulation=mfi > 50
        )
