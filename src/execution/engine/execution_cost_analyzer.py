from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ExecutionCosts:
    trading_fees: float
    slippage_cost: float
    impact_cost: float
    opportunity_cost: float
    total_cost: float
    cost_breakdown: Dict[str, float]

class ExecutionCostAnalyzer:
    def __init__(self, fee_structure: Dict = None):
        self.fee_structure = fee_structure or {
            'maker_fee': 0.0002,  # 0.02%
            'taker_fee': 0.0004   # 0.04%
        }
        
    async def analyze_costs(self, execution_data: Dict) -> ExecutionCosts:
        """실행 비용 분석"""
        fees = self._calculate_trading_fees(execution_data)
        slippage = self._calculate_slippage_cost(execution_data)
        impact = self._calculate_market_impact(execution_data)
        opportunity = self._calculate_opportunity_cost(execution_data)
        
        return ExecutionCosts(
            trading_fees=fees,
            slippage_cost=slippage,
            impact_cost=impact,
            opportunity_cost=opportunity,
            total_cost=fees + slippage + impact + opportunity,
            cost_breakdown=self._create_cost_breakdown(fees, slippage, impact, opportunity)
        )
