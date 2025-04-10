from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TradeFlowAnalysis:
    buy_pressure: float
    sell_pressure: float
    flow_imbalance: float
    trade_direction: str
    momentum_score: float

class TradeFlowAnalyzer:
    def __init__(self, analysis_config: Dict = None):
        self.config = analysis_config or {
            'flow_window': 100,
            'pressure_threshold': 0.6,
            'imbalance_threshold': 0.2
        }
        
    async def analyze_flow(self, trades: List[Dict]) -> TradeFlowAnalysis:
        """거래 흐름 분석"""
        buy_pressure = self._calculate_buy_pressure(trades)
        sell_pressure = self._calculate_sell_pressure(trades)
        imbalance = self._calculate_flow_imbalance(buy_pressure, sell_pressure)
        
        return TradeFlowAnalysis(
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            flow_imbalance=imbalance,
            trade_direction=self._determine_direction(imbalance),
            momentum_score=self._calculate_momentum(trades)
        )
