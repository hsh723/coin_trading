import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import optuna

@dataclass
class OptimizationResults:
    best_params: Dict
    optimization_history: List[Dict]
    param_importance: Dict[str, float]

class ParameterOptimizer:
    def __init__(self, param_space: Dict):
        self.param_space = param_space
        
    async def optimize_parameters(self, backtest_func, n_trials: int = 100) -> OptimizationResults:
        """전략 파라미터 최적화"""
        study = optuna.create_study(direction="maximize")
        
        for _ in range(n_trials):
            trial = study.ask()
            params = self._generate_parameters(trial)
            score = await backtest_func(params)
            study.tell(trial, score)
            
        return self._create_optimization_results(study)
