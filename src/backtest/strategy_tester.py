from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class TestResult:
    strategy_name: str
    total_returns: float
    drawdown: float
    trades: List[Dict]
    metrics: Dict[str, float]

class StrategyTester:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'initial_capital': 100000,
            'commission_rate': 0.001
        }
        
    async def test_strategy(self, strategy, market_data: pd.DataFrame) -> TestResult:
        """전략 백테스트 실행"""
        portfolio = self._initialize_portfolio()
        trades = []
        
        for timestamp, data in market_data.iterrows():
            signals = await strategy.generate_signals(data)
            if signals:
                execution = await self._execute_signals(signals, data, portfolio)
                trades.extend(execution['trades'])
                
        return TestResult(
            strategy_name=strategy.__class__.__name__,
            total_returns=self._calculate_returns(portfolio),
            drawdown=self._calculate_drawdown(portfolio),
            trades=trades,
            metrics=self._calculate_metrics(trades, portfolio)
        )
