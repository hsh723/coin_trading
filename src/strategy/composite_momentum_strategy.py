from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class CompositeMomentumSignal:
    price_momentum: float
    volume_momentum: float
    rsi_momentum: float
    composite_score: float
    signal_strength: str  # strong, medium, weak

class CompositeMomentumStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'price_window': 20,
            'volume_window': 10,
            'rsi_period': 14,
            'weight_price': 0.5,
            'weight_volume': 0.3,
            'weight_rsi': 0.2
        }
        
    async def generate_momentum_signal(self, market_data: pd.DataFrame) -> CompositeMomentumSignal:
        """복합 모멘텀 신호 생성"""
        price_mom = self._calculate_price_momentum(market_data)
        volume_mom = self._calculate_volume_momentum(market_data)
        rsi_mom = self._calculate_rsi_momentum(market_data)
        
        composite = (
            price_mom * self.config['weight_price'] +
            volume_mom * self.config['weight_volume'] +
            rsi_mom * self.config['weight_rsi']
        )
        
        return CompositeMomentumSignal(
            price_momentum=price_mom,
            volume_momentum=volume_mom,
            rsi_momentum=rsi_mom,
            composite_score=composite,
            signal_strength=self._determine_signal_strength(composite)
        )
