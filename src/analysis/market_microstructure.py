import numpy as np
import pandas as pd
from typing import Dict, Tuple

class MarketMicrostructureAnalyzer:
    def __init__(self, sampling_interval: str = '1min'):
        self.sampling_interval = sampling_interval
        
    def analyze_spread_dynamics(self, orderbook_data: pd.DataFrame) -> Dict[str, float]:
        """스프레드 역학 분석"""
        spreads = orderbook_data['ask'] - orderbook_data['bid']
        return {
            'effective_spread': self._calculate_effective_spread(spreads),
            'realized_spread': self._calculate_realized_spread(spreads),
            'price_impact': self._calculate_price_impact(orderbook_data)
        }
        
    def analyze_order_flow(self, trade_data: pd.DataFrame) -> Dict[str, float]:
        """주문 흐름 분석"""
        return {
            'buy_pressure': self._calculate_buy_pressure(trade_data),
            'trade_imbalance': self._calculate_trade_imbalance(trade_data),
            'order_to_trade_ratio': self._calculate_order_to_trade_ratio(trade_data)
        }
