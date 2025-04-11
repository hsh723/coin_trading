from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PositionSignal:
    size: float
    entry_type: str
    risk_factor: float
    confidence: float

class DynamicPositionManager:
    def __init__(self):
        self.config = {
            'max_position_size': 0.1,
            'risk_per_trade': 0.02,
            'leverage_limit': 3.0
        }
