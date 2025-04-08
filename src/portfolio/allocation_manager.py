import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class AllocationSignal:
    symbol: str
    target_weight: float
    current_weight: float
    action: str
    amount: float

class AllocationManager:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'rebalance_threshold': 0.05,
            'min_trade_size': 0.01
        }
        
    async def calculate_rebalance_signals(self, 
                                        current_positions: Dict,
                                        target_weights: Dict) -> List[AllocationSignal]:
        """리밸런싱 신호 생성"""
        signals = []
        total_value = self._calculate_total_value(current_positions)
        
        for symbol, target_weight in target_weights.items():
            current_weight = self._get_current_weight(symbol, current_positions, total_value)
            if abs(current_weight - target_weight) > self.config['rebalance_threshold']:
                signals.append(self._create_rebalance_signal(
                    symbol, target_weight, current_weight, total_value
                ))
                
        return signals
