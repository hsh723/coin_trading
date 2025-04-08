import numpy as np
from typing import Dict, List
from scipy import stats

class DriftDetector:
    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_stats = {}
        self.current_window = []

    def detect_drift(self, new_data: List[float]) -> Dict[str, bool]:
        """성능 저하 감지"""
        self.current_window.extend(new_data)
        if len(self.current_window) > self.window_size:
            self.current_window = self.current_window[-self.window_size:]

        drift_metrics = {
            'mean_shift': self._detect_mean_shift(),
            'volatility_change': self._detect_volatility_change(),
            'distribution_change': self._detect_distribution_change()
        }
        return drift_metrics
