from typing import Dict, List
import numpy as np
from ..analysis.volatility.volatility_analyzer import VolatilityAnalyzer

class AdaptiveGridStrategy:
    def __init__(self, config: Dict):
        self.grid_levels = config.get('grid_levels', 10)
        self.vol_analyzer = VolatilityAnalyzer()
        self.positions = {}
        
    def generate_grid_levels(self, current_price: float, volatility: float) -> Dict[str, List[float]]:
        """변동성 기반 그리드 레벨 생성"""
        grid_range = volatility * 2  # 2 표준편차 범위
        level_spread = grid_range / self.grid_levels
        
        buy_levels = [current_price * (1 - i * level_spread) 
                     for i in range(1, self.grid_levels + 1)]
        sell_levels = [current_price * (1 + i * level_spread) 
                      for i in range(1, self.grid_levels + 1)]
                      
        return {'buy_levels': buy_levels, 'sell_levels': sell_levels}
