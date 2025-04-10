from typing import Dict, List
from dataclasses import dataclass

@dataclass
class AlgorithmSelection:
    algorithm_id: str
    parameters: Dict
    expected_cost: float
    risk_score: float

class ExecutionAlgorithmSelector:
    def __init__(self, algorithms_config: Dict = None):
        self.config = algorithms_config or {
            'twap': {'time_window': 3600},
            'vwap': {'volume_profile': 'historical'},
            'impact': {'max_participation': 0.1}
        }
        
    async def select_algorithm(self, order: Dict, 
                             market_data: Dict) -> AlgorithmSelection:
        """실행 알고리즘 선택"""
        scores = {}
        for algo_id in self.config:
            scores[algo_id] = self._evaluate_algorithm(
                algo_id, order, market_data
            )
            
        best_algo = max(scores.items(), key=lambda x: x[1]['score'])[0]
        params = self._optimize_parameters(best_algo, order, market_data)
        
        return AlgorithmSelection(
            algorithm_id=best_algo,
            parameters=params,
            expected_cost=scores[best_algo]['cost'],
            risk_score=scores[best_algo]['risk']
        )
