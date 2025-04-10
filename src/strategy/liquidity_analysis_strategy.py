from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class LiquiditySignal:
    bid_liquidity: float
    ask_liquidity: float
    spread: float
    liquidity_score: float
    trade_recommendation: str

class LiquidityAnalysisStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'depth_levels': 10,
            'min_liquidity': 1.0,
            'spread_threshold': 0.002
        }
        
    async def analyze_liquidity(self, order_book: pd.DataFrame) -> LiquiditySignal:
        """유동성 분석 및 신호 생성"""
        bid_liquidity = self._calculate_bid_liquidity(order_book)
        ask_liquidity = self._calculate_ask_liquidity(order_book)
        current_spread = self._calculate_spread(order_book)
        
        return LiquiditySignal(
            bid_liquidity=bid_liquidity,
            ask_liquidity=ask_liquidity,
            spread=current_spread,
            liquidity_score=self._calculate_liquidity_score(bid_liquidity, ask_liquidity),
            trade_recommendation=self._generate_recommendation(bid_liquidity, ask_liquidity, current_spread)
        )
