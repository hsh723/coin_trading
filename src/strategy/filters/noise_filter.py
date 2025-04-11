import numpy as np
from typing import List, Dict

class NoiseFilter:
    def __init__(self, filter_config: Dict = None):
        self.config = filter_config or {
            'kalman_q': 0.001,
            'kalman_r': 0.1,
            'wavelet_level': 3
        }
        self.state = {'x': 0, 'p': 1}
        
    async def filter_signal(self, data: np.ndarray) -> Dict:
        kalman_filtered = self._apply_kalman(data)
        wavelet_filtered = self._apply_wavelet(data)
        
        return {
            'kalman': kalman_filtered,
            'wavelet': wavelet_filtered,
            'composite': self._combine_filters(kalman_filtered, wavelet_filtered)
        }
