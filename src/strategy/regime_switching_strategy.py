from typing import Dict, List
from dataclasses import dataclass
import numpy as np
from sklearn.mixture import GaussianMixture

@dataclass
class RegimeState:
    current_regime: str
    regime_probability: Dict[str, float]
    transition_matrix: np.ndarray
    optimal_strategy: str

class RegimeSwitchingStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'n_regimes': 3,
            'window_size': 100,
            'min_prob': 0.6
        }
        self.gmm = GaussianMixture(
            n_components=self.config['n_regimes'],
            covariance_type='full'
        )
        
    async def detect_regime(self, market_data: pd.DataFrame) -> RegimeState:
        """시장 레짐 감지 및 분석"""
        features = self._extract_features(market_data)
        regimes = self.gmm.fit_predict(features)
        
        return RegimeState(
            current_regime=self._identify_regime_type(regimes[-1]),
            regime_probability=self._calculate_regime_probabilities(),
            transition_matrix=self.gmm.transmat_,
            optimal_strategy=self._select_optimal_strategy(regimes[-1])
        )
