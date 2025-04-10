from typing import Dict, List
from dataclasses import dataclass

@dataclass
class OrderBookAnalysis:
    spread: float
    depth_score: float
    imbalance_ratio: float
    price_levels: Dict[str, List[float]]
    liquidity_distribution: Dict[str, float]

class OrderBookAnalyzer:
    def __init__(self, analysis_config: Dict = None):
        self.config = analysis_config or {
            'depth_levels': 10,
            'imbalance_threshold': 0.2,
            'min_liquidity': 1.0
        }
        
    async def analyze_order_book(self, 
                               order_book: Dict, 
                               current_price: float) -> OrderBookAnalysis:
        """오더북 분석"""
        bids = order_book['bids'][:self.config['depth_levels']]
        asks = order_book['asks'][:self.config['depth_levels']]
        
        return OrderBookAnalysis(
            spread=self._calculate_spread(bids[0][0], asks[0][0]),
            depth_score=self._calculate_depth_score(bids, asks),
            imbalance_ratio=self._calculate_imbalance(bids, asks),
            price_levels=self._analyze_price_levels(bids, asks),
            liquidity_distribution=self._analyze_liquidity(bids, asks)
        )
