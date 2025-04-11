import asyncio
from typing import Dict
import numpy as np

class ImbalanceDetector:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.imbalance_history = []
        
    async def detect_imbalances(self, market_data: Dict) -> Dict:
        """실시간 불균형 감지"""
        current_imbalance = await self._calculate_imbalance(market_data)
        is_significant = abs(current_imbalance) > self.threshold
        
        return {
            'imbalance_score': current_imbalance,
            'is_significant': is_significant,
            'direction': 'buy' if current_imbalance > 0 else 'sell',
            'magnitude': abs(current_imbalance)
        }
