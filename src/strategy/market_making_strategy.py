from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class MarketMakingSignal:
    bid_prices: List[float]
    ask_prices: List[float]
    spreads: List[float]
    inventory_target: float
    risk_metrics: Dict[str, float]

class MarketMakingStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'spread_multiplier': 1.5,
            'inventory_target': 0.0,
            'max_position': 1.0,
            'order_count': 5
        }
        
    async def generate_quotes(self, market_data: Dict) -> MarketMakingSignal:
        """호가 생성"""
        mid_price = (market_data['best_bid'] + market_data['best_ask']) / 2
        volatility = self._estimate_volatility(market_data)
        
        # 스프레드 계산 및 호가 생성
        spread = self._calculate_optimal_spread(volatility)
        quotes = self._generate_quote_ladder(mid_price, spread)
        
        return MarketMakingSignal(
            bid_prices=quotes['bids'],
            ask_prices=quotes['asks'],
            spreads=[spread] * self.config['order_count'],
            inventory_target=self._calculate_inventory_target(),
            risk_metrics=self._calculate_risk_metrics()
        )
