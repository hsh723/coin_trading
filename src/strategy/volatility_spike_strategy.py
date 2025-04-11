from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class VolatilitySpike:
    spike_detected: bool
    spike_magnitude: float
    price_direction: str
    risk_level: float
    action_recommendation: str

class VolatilitySpikeStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'spike_threshold': 2.0,
            'volatility_window': 20,
            'min_spike_size': 0.005
        }
        
    async def detect_spikes(self, market_data: pd.DataFrame) -> VolatilitySpike:
        """순간 변동성 스파이크 감지"""
        volatility = self._calculate_rolling_volatility(market_data)
        current_volatility = self._calculate_current_volatility(market_data)
        
        spike_magnitude = current_volatility / volatility.mean()
        spike_detected = spike_magnitude > self.config['spike_threshold']
        
        return VolatilitySpike(
            spike_detected=spike_detected,
            spike_magnitude=spike_magnitude,
            price_direction=self._determine_price_direction(market_data),
            risk_level=self._calculate_risk_level(spike_magnitude),
            action_recommendation=self._generate_recommendation(spike_detected, spike_magnitude)
        )
