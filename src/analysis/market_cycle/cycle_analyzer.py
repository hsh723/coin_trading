from typing import Dict, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class CyclePhase:
    name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration: int
    return_pct: float

class MarketCycleAnalyzer:
    def __init__(self, min_phase_length: int = 20):
        self.min_phase_length = min_phase_length
        self.phases = ['accumulation', 'markup', 'distribution', 'markdown']
        
    def identify_cycle_phase(self, data: pd.DataFrame) -> Dict:
        """현재 시장 사이클 단계 식별"""
        trend = self._calculate_trend(data)
        volume_trend = self._analyze_volume_pattern(data)
        momentum = self._calculate_momentum(data)
        
        return {
            'current_phase': self._determine_phase(trend, volume_trend, momentum),
            'phase_duration': self._calculate_phase_duration(data),
            'cycle_metrics': self._calculate_cycle_metrics(data)
        }
