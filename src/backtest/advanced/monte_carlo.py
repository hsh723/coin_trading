import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SimulationResult:
    confidence_intervals: Dict[str, tuple]
    max_drawdown_dist: np.ndarray
    final_value_dist: np.ndarray
    var_95: float
    cvar_95: float

class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 1000, confidence_level: float = 0.95):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level

    def run_simulation(self, returns: pd.Series, initial_value: float = 1.0) -> SimulationResult:
        """몬테카를로 시뮬레이션 실행"""
        paths = self._generate_paths(returns)
        cumulative_returns = self._calculate_cumulative_returns(paths, initial_value)
        
        return SimulationResult(
            confidence_intervals=self._calculate_confidence_intervals(cumulative_returns),
            max_drawdown_dist=self._calculate_max_drawdowns(cumulative_returns),
            final_value_dist=cumulative_returns[:, -1],
            var_95=self._calculate_var(cumulative_returns),
            cvar_95=self._calculate_cvar(cumulative_returns)
        )
