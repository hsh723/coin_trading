import numpy as np
from typing import Dict, List

class OrderFlowAnalyzer:
    def __init__(self):
        self.flow_metrics = {
            'buy_pressure': 0.0,
            'sell_pressure': 0.0,
            'net_flow': 0.0
        }
        
    async def analyze_flow(self, order_data: Dict) -> Dict:
        """실시간 주문 흐름 분석"""
        flow_analysis = await self._calculate_flow_metrics(order_data)
        imbalance = await self._detect_imbalances(flow_analysis)
        pressure = await self._calculate_pressure(flow_analysis)
        
        return {
            'flow_metrics': flow_analysis,
            'imbalances': imbalance,
            'pressure': pressure
        }
