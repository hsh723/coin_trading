import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ScenarioResult:
    returns: pd.Series
    drawdown: float
    sharpe_ratio: float
    var_95: float
    max_loss: float

class ScenarioAnalyzer:
    def __init__(self):
        self.scenarios = {}
        
    def add_scenario(self, name: str, market_conditions: Dict[str, float]):
        """시나리오 추가"""
        self.scenarios[name] = market_conditions
        
    def run_analysis(self, strategy, data: pd.DataFrame) -> Dict[str, ScenarioResult]:
        """시나리오별 분석 실행"""
        results = {}
        for scenario_name, conditions in self.scenarios.items():
            modified_data = self._apply_market_conditions(data, conditions)
            result = self._analyze_scenario(strategy, modified_data)
            results[scenario_name] = result
        return results
