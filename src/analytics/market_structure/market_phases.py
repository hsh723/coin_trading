from typing import Dict, List
import pandas as pd
from enum import Enum

class MarketPhase(Enum):
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"

class MarketStructureAnalyzer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    def identify_market_phase(self, market_data: pd.DataFrame) -> Dict:
        """시장 국면 식별"""
        volume_trend = self._analyze_volume_trend(market_data)
        price_structure = self._analyze_price_structure(market_data)
        momentum = self._calculate_momentum_indicators(market_data)
        
        return {
            'current_phase': self._determine_phase(price_structure, volume_trend),
            'confidence': self._calculate_phase_confidence(momentum),
            'duration': self._calculate_phase_duration(market_data)
        }
