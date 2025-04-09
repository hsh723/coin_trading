from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class TransactionCosts:
    fees: float
    slippage: float
    impact_cost: float
    opportunity_cost: float
    total_cost: float

class TransactionCostAnalyzer:
    def __init__(self, fee_structure: Dict = None):
        self.fee_structure = fee_structure or {
            'maker_fee': 0.001,
            'taker_fee': 0.002
        }
        
    async def analyze_costs(self, execution_data: Dict) -> TransactionCosts:
        """거래 비용 분석"""
        fees = self._calculate_fees(execution_data)
        slippage = self._calculate_slippage(execution_data)
        impact = self._calculate_market_impact(execution_data)
        opportunity = self._calculate_opportunity_cost(execution_data)
        
        return TransactionCosts(
            fees=fees,
            slippage=slippage,
            impact_cost=impact,
            opportunity_cost=opportunity,
            total_cost=fees + slippage + impact + opportunity
        )
