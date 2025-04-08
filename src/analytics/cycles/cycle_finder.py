import numpy as np
from scipy import signal
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class CycleComponent:
    period: int
    amplitude: float
    phase: float
    significance: float

class CycleFinder:
    def __init__(self, max_period: int = 252):
        self.max_period = max_period
        
    def find_dominant_cycles(self, data: np.ndarray, n_cycles: int = 3) -> List[CycleComponent]:
        """주요 시장 사이클 식별"""
        # FFT를 사용한 주기성 분석
        fft = np.fft.fft(data)
        frequencies = np.fft.fftfreq(len(data))
        
        # 주요 주기 성분 추출
        power_spectrum = np.abs(fft) ** 2
        dominant_freqs = self._find_dominant_frequencies(frequencies, power_spectrum, n_cycles)
        
        return [self._create_cycle_component(freq, fft, power_spectrum)
                for freq in dominant_freqs]
