from typing import Dict, List
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

@dataclass
class OptimizationResult:
    optimal_params: Dict[str, float]
    performance_metrics: Dict[str, float]
    optimization_path: List[Dict]
    convergence_info: Dict

class StrategyOptimizer:
    def __init__(self, optimization_config: Dict = None):
        self.config = optimization_config or {
            'max_iterations': 100,
            'objective': 'sharpe_ratio',
            'constraints': {'min_trades': 20}
        }
        
    async def optimize_strategy(self, 
                              strategy_class: type, 
                              market_data: Dict) -> OptimizationResult:
        """전략 파라미터 최적화"""
        param_bounds = self._get_parameter_bounds(strategy_class)
        
        def objective(params):
            strategy = strategy_class(self._create_param_dict(params))
            metrics = self._evaluate_strategy(strategy, market_data)
            return -metrics[self.config['objective']]
            
        result = minimize(
            objective,
            x0=self._get_initial_params(param_bounds),
            bounds=param_bounds,
            method='SLSQP'
        )
        
        return self._create_optimization_result(result)
