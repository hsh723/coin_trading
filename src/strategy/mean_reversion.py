from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from src.analysis.indicators.technical import TechnicalIndicators

class MeanReversionStrategy(BaseStrategy):
    """평균 회귀 기반 거래 전략"""
    
    def __init__(self, 
                 sma_period: int = 20,
                 std_dev: float = 2.0,
                 bb_period: int = 20,
                 bb_std: int = 2):
        super().__init__()
        self.sma_period = sma_period
        self.std_dev = std_dev
        self.bb_period = bb_period
        self.bb_std = bb_std
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """거래 신호 계산"""
        # 볼린저 밴드 계산
        bb = TechnicalIndicators.calculate_bollinger_bands(
            data['close'],
            self.bb_period,
            self.bb_std
        )
        
        # 이동평균 계산
        sma = data['close'].rolling(window=self.sma_period).mean()
        std = data['close'].rolling(window=self.sma_period).std()
        
        # 신호 생성
        signals = pd.DataFrame(index=data.index)
        signals['sma'] = sma
        signals['upper'] = sma + (std * self.std_dev)
        signals['lower'] = sma - (std * self.std_dev)
        signals['bb_middle'] = bb['middle']
        signals['bb_upper'] = bb['upper']
        signals['bb_lower'] = bb['lower']
        
        # 매수 조건: 가격이 하단 밴드 아래로 떨어지고, RSI가 과매도 구간
        rsi = TechnicalIndicators.calculate_rsi(data['close'])
        signals['buy_signal'] = (
            (data['close'] < signals['lower']) &
            (data['close'] < signals['bb_lower']) &
            (rsi < 30)
        )
        
        # 매도 조건: 가격이 상단 밴드 위로 올라가고, RSI가 과매수 구간
        signals['sell_signal'] = (
            (data['close'] > signals['upper']) &
            (data['close'] > signals['bb_upper']) &
            (rsi > 70)
        )
        
        return signals
    
    def execute_trade(self, 
                     data: pd.DataFrame,
                     position: Optional[float] = None,
                     balance: float = 0.0) -> Dict[str, float]:
        """거래 실행"""
        signals = self.calculate_signals(data)
        latest = signals.iloc[-1]
        
        if position is None or position == 0:
            if latest['buy_signal']:
                return {'action': 'buy', 'amount': balance}
        else:
            if latest['sell_signal']:
                return {'action': 'sell', 'amount': position}
                
        return {'action': 'hold', 'amount': 0.0} 