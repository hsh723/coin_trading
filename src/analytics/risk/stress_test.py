import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class StressTestResult:
    scenario_name: str
    max_loss: float
    var: float
    recovery_time: int
    impact_metrics: Dict[str, float]

class StressTester:
    def __init__(self, scenarios: Dict[str, Dict]):
        self.scenarios = scenarios
        
    def run_stress_test(self, portfolio: Dict[str, float], 
                       historical_data: pd.DataFrame) -> List[StressTestResult]:
        """스트레스 테스트 실행"""
        results = []
        for scenario_name, scenario_params in self.scenarios.items():
            modified_data = self._apply_scenario(historical_data, scenario_params)
            result = self._calculate_impact(portfolio, modified_data)
            results.append(StressTestResult(
                scenario_name=scenario_name,
                max_loss=result['max_loss'],
                var=result['var'],
                recovery_time=result['recovery_time'],
                impact_metrics=result['metrics']
            ))
        return results
