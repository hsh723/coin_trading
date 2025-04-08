import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime

class DynamicRebalancer:
    def __init__(self, 
                 rebalance_threshold: float = 0.05,
                 min_holding_period: int = 7):
        self.threshold = rebalance_threshold
        self.min_holding_period = min_holding_period
        self.last_rebalance = None
        
    def check_rebalance_needed(self, 
                              current_weights: Dict[str, float],
                              target_weights: Dict[str, float]) -> bool:
        """리밸런싱 필요 여부 확인"""
        if not self._check_holding_period():
            return False
            
        max_deviation = max(
            abs(current_weights.get(asset, 0) - target_weights.get(asset, 0))
            for asset in set(current_weights) | set(target_weights)
        )
        
        return max_deviation > self.threshold
