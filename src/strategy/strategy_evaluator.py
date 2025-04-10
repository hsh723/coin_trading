from typing import Dict, List
from dataclasses import dataclass

@dataclass
class EvaluationResults:
    strategy_id: str
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    consistency_score: float
    strategy_rating: str

class StrategyEvaluator:
    def __init__(self, evaluation_config: Dict = None):
        self.config = evaluation_config or {
            'min_trades': 30,
            'evaluation_period': '1M',
            'performance_weight': 0.6,
            'risk_weight': 0.4
        }
        
    async def evaluate_strategy(self, 
                              strategy_id: str,
                              trading_history: List[Dict]) -> EvaluationResults:
        """전략 성과 평가"""
        performance = self._analyze_performance(trading_history)
        risk = self._analyze_risk(trading_history)
        consistency = self._evaluate_consistency(trading_history)
        
        return EvaluationResults(
            strategy_id=strategy_id,
            performance_metrics=performance,
            risk_metrics=risk,
            consistency_score=consistency,
            strategy_rating=self._calculate_rating(performance, risk, consistency)
        )
