import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VolumeCycle:
    cycle_phase: str
    cycle_length: int
    cycle_amplitude: float
    next_peak_estimate: float

class VolumeCycleAnalyzer:
    def __init__(self, cycle_config: Dict = None):
        self.config = cycle_config or {
            'min_cycle_length': 5,
            'max_cycle_length': 50,
            'significance_level': 0.05
        }
        
    async def analyze_cycles(self, volume_data: np.ndarray) -> VolumeCycle:
        """거래량 주기 분석"""
        cycle_data = self._detect_cycles(volume_data)
        phase = self._determine_cycle_phase(volume_data)
        
        return VolumeCycle(
            cycle_phase=phase,
            cycle_length=self._calculate_cycle_length(cycle_data),
            cycle_amplitude=self._calculate_amplitude(cycle_data),
            next_peak_estimate=self._estimate_next_peak(cycle_data)
        )
