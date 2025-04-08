from typing import List, Dict
import numpy as np
import pandas as pd

class ScenarioGenerator:
    def __init__(self, num_scenarios: int = 1000):
        self.num_scenarios = num_scenarios
        
    def generate_market_scenarios(self, historical_data: pd.DataFrame) -> List[pd.DataFrame]:
        """시장 시나리오 생성"""
        returns = historical_data.pct_change().dropna()
        cov_matrix = returns.cov()
        mean_returns = returns.mean()
        
        scenarios = []
        for _ in range(self.num_scenarios):
            scenario = self._generate_single_scenario(
                mean_returns,
                cov_matrix,
                len(historical_data)
            )
            scenarios.append(scenario)
            
        return scenarios
