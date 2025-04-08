from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, exchange, risk_manager):
        self.exchange = exchange
        self.risk_manager = risk_manager
        self.positions = {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """거래 신호 생성"""
        pass

    def calculate_position_size(self, signal: str, price: float) -> float:
        """포지션 크기 계산"""
        return self.risk_manager.calculate_position_size(
            signal=signal,
            current_price=price
        )

    def execute_strategy(self, data: pd.DataFrame) -> List[Dict]:
        """전략 실행"""
        signals = self.generate_signals(data)
        trades = []
        
        for symbol, signal in signals.items():
            if signal in ['BUY', 'SELL']:
                size = self.calculate_position_size(signal, data['close'].iloc[-1])
                trade = self.execute_trade(symbol, signal, size)
                trades.append(trade)
                
        return trades

    def execute_trade(self, symbol: str, signal: str, size: float) -> Dict:
        """거래 실행"""
        # 구현...
