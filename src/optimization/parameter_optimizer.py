from typing import Dict, List
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from ..backtest.engine import BacktestEngine

class ParameterOptimizer:
    def __init__(self, strategy_class, data: pd.DataFrame):
        self.strategy_class = strategy_class
        self.data = data
        self.best_params = None
        self.best_score = float('-inf')
        
    def optimize(self, param_ranges: Dict, metric: str = 'sharpe_ratio') -> Dict:
        """그리드 서치를 통한 파라미터 최적화"""
        param_combinations = self._generate_param_combinations(param_ranges)
        
        for params in param_combinations:
            strategy = self.strategy_class(params)
            backtest = BacktestEngine(strategy)
            results = backtest.run(self.data)
            
            score = results['metrics'][metric]
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                
        return {
            'best_params': self.best_params,
            'best_score': self.best_score
        }
