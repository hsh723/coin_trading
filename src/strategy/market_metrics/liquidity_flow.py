from dataclasses import dataclass
import pandas as pd

@dataclass
class LiquidityFlow:
    net_flow: float
    buy_pressure: float
    sell_pressure: float
    flow_strength: float
    imbalance_ratio: float

class LiquidityFlowAnalyzer:
    def __init__(self):
        self.metrics = {
            'flow_window': 50,
            'pressure_threshold': 0.6,
            'strength_multiplier': 1.5
        }
