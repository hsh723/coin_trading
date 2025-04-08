from typing import Dict, List
import numpy as np
import pandas as pd

class PortfolioManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades_history: List[Dict] = []

    def calculate_allocation(self, signals: Dict[str, float]) -> Dict[str, float]:
        """포지션 크기 계산"""
        total_signal = sum(abs(v) for v in signals.values())
        if total_signal == 0:
            return {}
            
        allocations = {
            symbol: (signal / total_signal) * self.current_capital
            for symbol, signal in signals.items()
        }
        return allocations

    def update_positions(self, trades: List[Dict]) -> None:
        """포지션 업데이트"""
        for trade in trades:
            symbol = trade['symbol']
            if trade['action'] == 'OPEN':
                self.positions[symbol] = {
                    'amount': trade['amount'],
                    'entry_price': trade['price']
                }
            elif trade['action'] == 'CLOSE':
                if symbol in self.positions:
                    del self.positions[symbol]
