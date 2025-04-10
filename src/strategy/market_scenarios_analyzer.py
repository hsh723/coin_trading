from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class MarketScenario:
    scenario_type: str
    probability: float
    risk_impact: Dict[str, float]
    suggested_actions: List[str]

class MarketScenariosAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_probability': 0.2,
            'lookback_period': 30,
            'scenario_types': ['trend_reversal', 'breakout', 'consolidation']
        }
        
    async def analyze_scenarios(self, market_data: pd.DataFrame) -> List[MarketScenario]:
        """시장 시나리오 분석"""
        scenarios = []
        
        for scenario_type in self.config['scenario_types']:
            probability = self._calculate_scenario_probability(market_data, scenario_type)
            if probability > self.config['min_probability']:
                scenarios.append(
                    self._create_scenario(scenario_type, probability, market_data)
                )
                
        return sorted(scenarios, key=lambda x: x.probability, reverse=True)
