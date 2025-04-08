import pandas as pd
import numpy as np
from typing import Dict

class VolatilityAdjuster:
    def __init__(self, config: Dict):
        self.base_size = config.get('base_order_size', 1.0)
        self.vol_window = config.get('volatility_window', 24)
        self.vol_target = config.get('volatility_target', 0.02)
        
    def adjust_execution_params(self, market_data: pd.DataFrame) -> Dict:
        """변동성 기반 실행 파라미터 조정"""
        current_vol = self._calculate_current_volatility(market_data)
        vol_ratio = current_vol / self.vol_target
        
        return {
            'order_size': self._adjust_size(vol_ratio),
            'execution_speed': self._adjust_speed(vol_ratio),
            'slippage_tolerance': self._adjust_slippage(vol_ratio)
        }
