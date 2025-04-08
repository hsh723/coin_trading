import numpy as np
import pandas as pd
from typing import Dict, List
from ..risk.manager import RiskManager

class MarketMaker:
    def __init__(self, config: Dict):
        self.spread_multiplier = config.get('spread_multiplier', 1.5)
        self.min_spread = config.get('min_spread', 0.001)
        self.risk_manager = RiskManager(config.get('risk_config', {}))
        
    def calculate_quotes(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """호가 계산"""
        volatility = self._calculate_volatility(market_data)
        spread = max(self.min_spread, volatility * self.spread_multiplier)
        mid_price = market_data['close'].iloc[-1]
        
        return {
            'bid': mid_price * (1 - spread/2),
            'ask': mid_price * (1 + spread/2),
            'spread': spread
        }
