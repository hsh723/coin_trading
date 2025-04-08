from typing import Dict, List
import optuna
import numpy as np
from ..backtest.engine import BacktestEngine

class ParameterTuner:
    def __init__(self, strategy_class, data: pd.DataFrame):
        self.strategy_class = strategy_class
        self.data = data
        self.study = None
        
    def optimize_parameters(self, param_space: Dict, n_trials: int = 100) -> Dict:
        """파라미터 최적화"""
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(
            lambda trial: self._objective(trial, param_space),
            n_trials=n_trials
        )
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value
        }
