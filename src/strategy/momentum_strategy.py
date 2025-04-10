from typing import Dict
from .base_strategy import BaseStrategy, StrategyResult

class MomentumStrategy(BaseStrategy):
    def __init__(self, params: Dict):
        super().__init__()
        self.tech_analyzer = TechnicalAnalyzer()
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.rsi_oversold = params.get('rsi_oversold', 30)
    
    async def generate_signal(self, market_data: Dict) -> StrategyResult:
        """모멘텀 기반 신호 생성"""
        momentum = self._calculate_momentum(market_data)
        signal = 'buy' if momentum > self.config['momentum_threshold'] else 'sell'
        
        return StrategyResult(
            signal=signal,
            confidence=abs(momentum),
            params={'momentum': momentum},
            metadata={'timeframe': '1h'}
        )
