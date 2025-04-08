from typing import Dict, List
import optuna
import numpy as np
from ..backtest.engine import BacktestEngine

class BacktestOptimizer:
    def __init__(self, strategy_class, data: pd.DataFrame):
        self.strategy_class = strategy_class
        self.data = data
        self.best_params = None
        
    def optimize(self, param_ranges: Dict, n_trials: int = 100) -> Dict:
        """하이퍼파라미터 최적화"""
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, param_ranges),
            n_trials=n_trials
        )
        
        self.best_params = study.best_params
        return {
            'best_params': study.best_params,
            'best_value': study.best_value
        }
