from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class ATRStopSignal:
    stop_price: float
    atr_value: float
    multiplier: float
    stop_updated: bool

class ATRTrailingStop:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'minimum_stop': 0.01
        }
        
    async def calculate_stop(self, market_data: pd.DataFrame, position_side: str) -> ATRStopSignal:
        """ATR 기반 스탑로스 계산"""
        atr = self._calculate_atr(market_data)
        current_price = market_data['close'].iloc[-1]
        stop_distance = atr * self.config['atr_multiplier']
        
        stop_price = (current_price - stop_distance if position_side == 'long' 
                     else current_price + stop_distance)
                     
        return ATRStopSignal(
            stop_price=stop_price,
            atr_value=atr,
            multiplier=self.config['atr_multiplier'],
            stop_updated=True
        )
