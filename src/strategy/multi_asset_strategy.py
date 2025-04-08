from typing import Dict, List
import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..risk.portfolio_risk import PortfolioRiskManager

class MultiAssetStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.risk_manager = PortfolioRiskManager(config.get('risk_params', {}))
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.max_position_per_asset = config.get('max_position_per_asset', 0.2)
        
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """다중 자산 신호 생성"""
        signals = {}
        correlation_matrix = self._calculate_correlation_matrix(market_data)
        
        for symbol, data in market_data.items():
            if self._check_correlation_limits(symbol, correlation_matrix):
                signal = await self._analyze_single_asset(data)
                signals[symbol] = signal
                
        return self._adjust_position_sizes(signals)
