import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class CumulativeVolumeAnalysis:
    cvd_trend: str
    buying_pressure: float
    selling_pressure: float
    accumulation_zones: List[float]

class CumulativeVolumeAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'pressure_threshold': 0.6,
            'zone_threshold': 0.1
        }
        
    async def analyze_cvd(self, price_data: np.ndarray, volume_data: np.ndarray) -> CumulativeVolumeAnalysis:
        """누적 거래량 분포 분석"""
        cvd = self._calculate_cvd(price_data, volume_data)
        buying_pressure = self._calculate_buying_pressure(cvd)
        selling_pressure = self._calculate_selling_pressure(cvd)
        
        return CumulativeVolumeAnalysis(
            cvd_trend=self._determine_cvd_trend(cvd),
            buying_pressure=buying_pressure,
            selling_pressure=selling_pressure,
            accumulation_zones=self._find_accumulation_zones(cvd)
        )
