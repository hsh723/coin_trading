import optuna
from typing import Dict, Callable
import numpy as np
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    best_params: Dict
    best_score: float
    study_trials: List

class HyperparameterTuner:
    def __init__(self, objective_func: Callable, param_space: Dict):
        self.objective_func = objective_func
        self.param_space = param_space
        self.study = None
        
    def optimize(self, n_trials: int = 100) -> OptimizationResult:
        """하이퍼파라미터 최적화 실행"""
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self._objective, n_trials=n_trials)
        
        return OptimizationResult(
            best_params=self.study.best_params,
            best_score=self.study.best_value,
            study_trials=self.study.trials
        )
