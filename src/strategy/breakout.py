"""
브레이크아웃 전략 클래스
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from src.strategy.base_strategy import BaseStrategy, StrategyResult
from src.analysis.indicators.technical import TechnicalIndicators
from src.utils.logger import get_logger

class BreakoutStrategy(BaseStrategy):
    """브레이크아웃 전략 클래스"""
    
    def __init__(self,
                 lookback_period: int = 20,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 volume_threshold: float = 1.5):
        super().__init__()
        self.lookback_period = lookback_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.volume_threshold = volume_threshold
        
    def initialize(self, data: pd.DataFrame) -> None:
        """전략 초기화"""
        self._state = {
            'last_signal': None,
            'position': 0,
            'entry_price': 0.0
        }
        
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """ATR 계산"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """시장 분석"""
        # 고가/저가 저항/지지선
        high = data['high'].rolling(window=self.lookback_period).max()
        low = data['low'].rolling(window=self.lookback_period).min()
        
        # ATR 계산
        atr = self.calculate_atr(data)
        
        # 거래량 이동평균
        volume_ma = data['volume'].rolling(window=self.lookback_period).mean()
        
        return {
            'high': high,
            'low': low,
            'atr': atr,
            'volume_ma': volume_ma
        }
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """매매 신호 생성"""
        analysis = self.analyze(data)
        latest = data.iloc[-1]
        
        # 가격이 고가 저항선을 돌파하고 거래량이 평균 이상
        buy_signal = (
            (latest['close'] > analysis['high'].iloc[-2]) &
            (latest['volume'] > analysis['volume_ma'].iloc[-1] * self.volume_threshold)
        )
        
        # 가격이 저가 지지선을 하향 돌파하고 거래량이 평균 이상
        sell_signal = (
            (latest['close'] < analysis['low'].iloc[-2]) &
            (latest['volume'] > analysis['volume_ma'].iloc[-1] * self.volume_threshold)
        )
        
        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'price': latest['close'],
            'analysis': {
                'high': analysis['high'].iloc[-1],
                'low': analysis['low'].iloc[-1],
                'atr': analysis['atr'].iloc[-1],
                'volume_ma': analysis['volume_ma'].iloc[-1]
            }
        }
        
    def execute(self, data: pd.DataFrame, position: Optional[float] = None) -> Dict[str, Any]:
        """매매 실행"""
        signals = self.generate_signals(data)
        
        if position is None or position == 0:
            if signals['buy_signal']:
                return {'action': 'buy', 'amount': 1.0}
        else:
            if signals['sell_signal']:
                return {'action': 'sell', 'amount': position}
                
        return {'action': 'hold', 'amount': 0.0}
        
    def update(self, data: pd.DataFrame) -> None:
        """전략 상태 업데이트"""
        signals = self.generate_signals(data)
        self._state['last_signal'] = signals
        
    def get_state(self) -> Dict[str, Any]:
        """전략 상태 반환"""
        return self._state
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """전략 상태 설정"""
        self._state = state 

    async def generate_signal(self, market_data: Dict) -> StrategyResult:
        """전략 신호 생성"""
        data = pd.DataFrame(market_data)
        signals = self.generate_signals(data)
        
        if signals['buy_signal']:
            return StrategyResult(
                signal='buy',
                confidence=0.8,
                params=self.config,
                metadata={'analysis': signals['analysis']}
            )
        elif signals['sell_signal']:
            return StrategyResult(
                signal='sell',
                confidence=0.8,
                params=self.config,
                metadata={'analysis': signals['analysis']}
            )
        else:
            return StrategyResult(
                signal='hold',
                confidence=0.5,
                params=self.config,
                metadata={'analysis': signals['analysis']}
            ) 