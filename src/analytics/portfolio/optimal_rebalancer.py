import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RebalanceAction:
    symbol: str
    current_weight: float
    target_weight: float
    action: str
    amount: float

class OptimalRebalancer:
    def __init__(self, config: Dict):
        self.threshold = config.get('rebalance_threshold', 0.05)
        self.min_trade_size = config.get('min_trade_size', 0.01)
        
    def calculate_rebalance_actions(self, 
                                  current_weights: Dict[str, float],
                                  target_weights: Dict[str, float],
                                  portfolio_value: float) -> List[RebalanceAction]:
        """최적 리밸런싱 액션 계산"""
        actions = []
        for symbol in set(current_weights) | set(target_weights):
            curr_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            
            if abs(curr_weight - target_weight) > self.threshold:
                actions.append(self._create_rebalance_action(
                    symbol, curr_weight, target_weight, portfolio_value
                ))
        
        return actions
