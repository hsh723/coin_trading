"""
평균회귀 전략 클래스
"""

from typing import Dict, Any, Optional
import pandas as pd
from src.strategy.base_strategy import BaseStrategy, StrategyResult
from src.analysis.indicators.technical import TechnicalIndicators
from src.utils.logger import get_logger

class MeanReversionStrategy(BaseStrategy):
    """평균회귀 전략 클래스"""
    
    def __init__(self,
                 sma_period: int = 20,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30):
        super().__init__()
        self.sma_period = sma_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
    def initialize(self, data: pd.DataFrame) -> None:
        """전략 초기화"""
        self._state = {
            'last_signal': None,
            'position': 0,
            'entry_price': 0.0
        }
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """시장 분석"""
        # SMA 계산
        sma = TechnicalIndicators.calculate_sma(data['close'], self.sma_period)
        
        # 표준편차 계산
        std = data['close'].rolling(window=self.bb_period).std()
        
        # 볼린저 밴드 계산
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(
            data['close'],
            self.bb_period,
            self.bb_std
        )
        
        # RSI 계산
        rsi = TechnicalIndicators.calculate_rsi(data['close'], self.rsi_period)
        
        # Z-score 계산
        zscore = (data['close'] - sma) / std
        
        return {
            'sma': sma,
            'std': std,
            'zscore': zscore,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'rsi': rsi
        }
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """매매 신호 생성"""
        analysis = self.analyze(data)
        latest = data.iloc[-1]
        latest_close = latest['close']
        
        # 매수 조건: 가격이 볼린저 밴드 하단을 하향 돌파하고 RSI가 과매도 구간
        buy_signal = (
            (latest_close < analysis['bb_lower'].iloc[-1]) &
            (analysis['rsi'].iloc[-1] < self.rsi_oversold)
        )
        
        # 매도 조건: 가격이 볼린저 밴드 상단을 상향 돌파하고 RSI가 과매수 구간
        sell_signal = (
            (latest_close > analysis['bb_upper'].iloc[-1]) &
            (analysis['rsi'].iloc[-1] > self.rsi_overbought)
        )
        
        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'price': latest_close,
            'analysis': {
                'sma': analysis['sma'].iloc[-1],
                'bb_upper': analysis['bb_upper'].iloc[-1],
                'bb_middle': analysis['bb_middle'].iloc[-1],
                'bb_lower': analysis['bb_lower'].iloc[-1],
                'rsi': analysis['rsi'].iloc[-1]
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