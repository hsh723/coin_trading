from typing import Dict, List
import optuna
import pandas as pd
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    best_params: Dict
    best_score: float
    optimization_history: List[Dict]
    parameter_importance: Dict[str, float]

class StrategyOptimizer:
    def __init__(self, strategy_class, param_space: Dict):
        self.strategy_class = strategy_class
        self.param_space = param_space
        
    async def optimize_strategy(self, market_data: pd.DataFrame, 
                              n_trials: int = 100) -> OptimizationResult:
        """전략 파라미터 최적화"""
        study = optuna.create_study(direction="maximize")
        
        for _ in range(n_trials):
            trial = study.ask()
            params = self._generate_params(trial)
            score = await self._evaluate_params(params, market_data)
            study.tell(trial, score)
            
        return self._create_optimization_result(study)
