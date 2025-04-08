from typing import Dict, List
import pandas as pd
from .base import BaseStrategy

class MultiFactorStrategy(BaseStrategy):
    def __init__(self, factors: List[Dict]):
        super().__init__()
        self.factors = factors
        self.weights = self._initialize_factor_weights()
        
    async def calculate_factor_scores(self, data: pd.DataFrame) -> Dict[str, float]:
        """요인별 점수 계산"""
        scores = {}
        for factor in self.factors:
            factor_name = factor['name']
            factor_func = getattr(self, f"calculate_{factor_name}_score")
            scores[factor_name] = await factor_func(data)
            
        return self._combine_factor_scores(scores)
