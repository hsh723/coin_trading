from typing import Dict, List
import pandas as pd
from .base import BaseStrategy

class ArbitrageStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.min_spread = config.get('min_spread', 0.001)  # 최소 스프레드
        self.max_exposure = config.get('max_exposure', 100000)  # 최대 노출도
        
    async def find_opportunities(self, prices: Dict[str, float]) -> List[Dict]:
        """차익거래 기회 탐색"""
        opportunities = []
        exchanges = list(prices.keys())
        
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                spread = prices[exchanges[j]] / prices[exchanges[i]] - 1
                if spread > self.min_spread:
                    opportunities.append({
                        'buy_exchange': exchanges[i],
                        'sell_exchange': exchanges[j],
                        'spread': spread
                    })
        return opportunities
