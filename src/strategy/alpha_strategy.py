from typing import Dict, List
import pandas as pd
from .base import BaseStrategy
from ..analysis.technical import TechnicalAnalyzer
from ..analysis.market_regime import MarketRegimeAnalyzer

class AlphaStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.tech_analyzer = TechnicalAnalyzer()
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.alpha_threshold = config.get('alpha_threshold', 0.002)
        
    def calculate_alpha(self, asset_data: pd.DataFrame, benchmark_data: pd.DataFrame) -> float:
        """알파 계산"""
        asset_returns = asset_data['close'].pct_change()
        benchmark_returns = benchmark_data['close'].pct_change()
        
        beta = self._calculate_beta(asset_returns, benchmark_returns)
        alpha = asset_returns.mean() - (self.risk_free_rate + beta * (benchmark_returns.mean() - self.risk_free_rate))
        return alpha
