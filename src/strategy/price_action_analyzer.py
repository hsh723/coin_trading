from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class PriceAction:
    action_type: str  # rejection, breakout, consolidation
    level_type: str  # support, resistance
    strength: float
    key_levels: List[float]
    volume_confirmation: bool

class PriceActionAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'lookback_period': 20,
            'volume_threshold': 1.5,
            'level_tolerance': 0.002
        }
        
    async def analyze_price_action(self, market_data: pd.DataFrame) -> PriceAction:
        """가격 행동 분석"""
        current_price = market_data['close'].iloc[-1]
        levels = self._identify_key_levels(market_data)
        
        return PriceAction(
            action_type=self._determine_action_type(current_price, levels),
            level_type=self._determine_level_type(current_price, levels),
            strength=self._calculate_level_strength(market_data),
            key_levels=levels,
            volume_confirmation=self._check_volume_confirmation(market_data)
        )
