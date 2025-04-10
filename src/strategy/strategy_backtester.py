from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class BacktestResults:
    total_returns: float
    trades: List[Dict]
    metrics: Dict[str, float]
    equity_curve: pd.Series

class StrategyBacktester:
    def __init__(self, backtest_config: Dict = None):
        self.config = backtest_config or {
            'initial_capital': 100000,
            'commission': 0.001,  # 0.1%
            'slippage': 0.001    # 0.1%
        }
        
    async def run_backtest(self, 
                          strategy: any, 
                          market_data: pd.DataFrame) -> BacktestResults:
        """전략 백테스트 실행"""
        capital = self.config['initial_capital']
        trades = []
        equity_curve = []
        
        for timestamp, data in market_data.iterrows():
            signals = await strategy.generate_signals(data)
            if signals:
                trade_result = self._execute_trade(signals, data, capital)
                trades.append(trade_result)
                capital += trade_result['pnl']
            equity_curve.append(capital)
            
        return self._generate_backtest_results(trades, equity_curve)
