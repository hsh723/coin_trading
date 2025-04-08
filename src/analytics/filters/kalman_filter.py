import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class KalmanState:
    state: np.ndarray
    covariance: np.ndarray
    gain: np.ndarray

class KalmanFilter:
    def __init__(self, n_states: int = 2):
        self.n_states = n_states
        self.state = np.zeros(n_states)
        self.covariance = np.eye(n_states)
        
    def update(self, measurement: float, 
              timestamp: float = None) -> KalmanState:
        """칼만 필터 상태 업데이트"""
        # 예측 단계
        predicted_state = self._predict_state()
        predicted_covariance = self._predict_covariance()
        
        # 업데이트 단계
        kalman_gain = self._calculate_gain(predicted_covariance)
        self.state = self._update_state(predicted_state, 
                                      measurement, 
                                      kalman_gain)
        self.covariance = self._update_covariance(predicted_covariance, 
                                                 kalman_gain)
        
        return KalmanState(
            state=self.state,
            covariance=self.covariance,
            gain=kalman_gain
        )
