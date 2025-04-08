import numpy as np
from typing import Dict, List
import optuna
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    best_params: Dict
    best_score: float
    optimization_history: List
    
class AdaptiveOptimizer:
    def __init__(self, objective_func, param_space: Dict):
        self.objective_func = objective_func
        self.param_space = param_space
        self.study = None
        
    def optimize(self, n_trials: int = 100) -> OptimizationResult:
        """적응형 파라미터 최적화"""
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self._objective, n_trials=n_trials)
        
        return OptimizationResult(
            best_params=self.study.best_params,
            best_score=self.study.best_value,
            optimization_history=self.study.trials
        )
