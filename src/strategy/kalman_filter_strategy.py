from typing import Dict, List
from dataclasses import dataclass
import numpy as np
from scipy.linalg import inv

@dataclass
class KalmanEstimate:
    state: np.ndarray
    covariance: np.ndarray
    prediction: float
    confidence: float

class KalmanFilterStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'state_dim': 2,
            'observation_dim': 1,
            'process_variance': 1e-4,
            'observation_variance': 1e-2
        }
        self.state = np.zeros(self.config['state_dim'])
        self.covariance = np.eye(self.config['state_dim'])
        
    async def update(self, price: float) -> KalmanEstimate:
        """칼만 필터 업데이트 및 예측"""
        # 상태 예측
        predicted_state = self._predict_state()
        predicted_covar = self._predict_covariance()
        
        # 측정값 업데이트
        kalman_gain = self._calculate_kalman_gain(predicted_covar)
        self.state = self._update_state(predicted_state, kalman_gain, price)
        self.covariance = self._update_covariance(predicted_covar, kalman_gain)
        
        return KalmanEstimate(
            state=self.state,
            covariance=self.covariance,
            prediction=self._get_prediction(),
            confidence=self._calculate_confidence()
        )
