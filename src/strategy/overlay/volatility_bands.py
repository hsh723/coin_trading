from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class VolatilityBands:
    upper_band: List[float]
    lower_band: List[float]
    middle_band: List[float]
    volatility_state: str
    band_width: float

class VolatilityBandsAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = {
            'std_multiplier': 2.0,
            'lookback_period': 20,
            'min_band_width': 0.005
        }
