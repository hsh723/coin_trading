from typing import Dict, Callable
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class BayesianOptimizer:
    def __init__(self, param_space: Dict, objective_func: Callable):
        self.param_space = param_space
        self.objective_func = objective_func
        self.gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=10
        )
        
    def optimize(self, n_iterations: int = 50) -> Dict:
        """베이지안 최적화 실행"""
        X_samples = []
        y_samples = []
        
        for i in range(n_iterations):
            next_point = self._suggest_next_point(X_samples, y_samples)
            score = self.objective_func(next_point)
            
            X_samples.append(next_point)
            y_samples.append(score)
