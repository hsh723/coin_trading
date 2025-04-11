from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ExitSignal:
    exit_price: float
    exit_type: str
    urgency: float
    stop_adjustment: float

class DynamicExitStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'trailing_factor': 0.02,
            'volatility_adjust': True,
            'profit_lock': 0.5
        }
        
    async def calculate_exit(self, position_data: Dict, market_data: Dict) -> ExitSignal:
        """동적 청산 가격 계산"""
        return ExitSignal(
            exit_price=self._calculate_optimal_exit(position_data, market_data),
            exit_type=self._determine_exit_type(position_data),
            urgency=self._calculate_exit_urgency(market_data),
            stop_adjustment=self._calculate_stop_adjustment(market_data)
        )
