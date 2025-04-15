"""
기본 전략 모듈
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.utils.logger import setup_logger
from src.indicators.basic import TechnicalIndicators
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class TrendType(Enum):
    """추세 유형"""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"
    
@dataclass
class TrendInfo:
    """추세 정보"""
    type: TrendType
    strength: float
    start_price: float
    current_price: float
    duration: int
    
@dataclass
class FibonacciLevels:
    """피보나치 레벨"""
    level_0: float
    level_236: float
    level_382: float
    level_500: float
    level_618: float
    level_786: float
    level_1000: float
    
@dataclass
class StochasticSignal:
    """스토캐스틱 신호"""
    k_value: float
    d_value: float
    is_overbought: bool
    is_oversold: bool
    
@dataclass
class TrendlineInfo:
    """추세선 정보"""
    slope: float
    intercept: float
    r_squared: float
    support_points: List[float]
    resistance_points: List[float]

class BaseStrategy(ABC):
    """
    기본 전략 클래스
    모든 전략의 기본이 되는 추상 클래스
    """
    
    def __init__(self):
        """
        전략 초기화
        """
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators()
        self.name = self.__class__.__name__
        
    @abstractmethod
    async def generate_signal(
        self,
        market_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        거래 신호 생성
        
        Args:
            market_data (pd.DataFrame): 시장 데이터
            
        Returns:
            Optional[Dict[str, Any]]: 거래 신호
        """
        pass
        
    def calculate_position_size(
        self,
        capital: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """
        포지션 크기 계산
        
        Args:
            capital (float): 자본금
            risk_per_trade (float): 거래당 위험 비율
            entry_price (float): 진입 가격
            stop_loss (float): 손절가
            
        Returns:
            float: 포지션 크기
        """
        try:
            risk_amount = capital * risk_per_trade
            price_risk = abs(entry_price - stop_loss)
            position_size = risk_amount / price_risk
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {str(e)}")
            return 0.0
            
    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> float:
        """
        위험 대비 수익 비율 계산
        
        Args:
            entry_price (float): 진입 가격
            stop_loss (float): 손절가
            take_profit (float): 익절가
            
        Returns:
            float: 위험 대비 수익 비율
        """
        try:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            return reward / risk if risk > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"위험 대비 수익 비율 계산 실패: {str(e)}")
            return 0.0
            
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        거래 신호 유효성 검사
        
        Args:
            signal (Dict[str, Any]): 거래 신호
            
        Returns:
            bool: 유효성 여부
        """
        try:
            required_fields = [
                'symbol',
                'side',
                'type',
                'price',
                'stop_loss',
                'take_profit'
            ]
            
            # 필수 필드 확인
            if not all(field in signal for field in required_fields):
                self.logger.error("거래 신호에 필수 필드가 누락되었습니다")
                return False
                
            # 가격 유효성 확인
            if signal['price'] <= 0:
                self.logger.error("진입 가격이 유효하지 않습니다")
                return False
                
            # 매수 신호 검증
            if signal['side'] == 'buy':
                if signal['stop_loss'] >= signal['price']:
                    self.logger.error("매수 손절가가 진입 가격보다 높습니다")
                    return False
                if signal['take_profit'] <= signal['price']:
                    self.logger.error("매수 익절가가 진입 가격보다 낮습니다")
                    return False
                    
            # 매도 신호 검증
            elif signal['side'] == 'sell':
                if signal['stop_loss'] <= signal['price']:
                    self.logger.error("매도 손절가가 진입 가격보다 낮습니다")
                    return False
                if signal['take_profit'] >= signal['price']:
                    self.logger.error("매도 익절가가 진입 가격보다 높습니다")
                    return False
                    
            else:
                self.logger.error("거래 방향이 유효하지 않습니다")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"거래 신호 유효성 검사 실패: {str(e)}")
            return False
            
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr: float,
        multiplier: float = 2.0
    ) -> float:
        """
        손절가 계산
        
        Args:
            entry_price (float): 진입 가격
            side (str): 거래 방향
            atr (float): ATR
            multiplier (float): ATR 승수
            
        Returns:
            float: 손절가
        """
        try:
            stop_distance = atr * multiplier
            
            if side == 'buy':
                return entry_price - stop_distance
            elif side == 'sell':
                return entry_price + stop_distance
            else:
                self.logger.error("거래 방향이 유효하지 않습니다")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"손절가 계산 실패: {str(e)}")
            return 0.0
            
    def calculate_take_profit(
        self,
        entry_price: float,
        side: str,
        risk_reward: float,
        stop_loss: float
    ) -> float:
        """
        익절가 계산
        
        Args:
            entry_price (float): 진입 가격
            side (str): 거래 방향
            risk_reward (float): 위험 대비 수익 비율
            stop_loss (float): 손절가
            
        Returns:
            float: 익절가
        """
        try:
            risk = abs(entry_price - stop_loss)
            reward = risk * risk_reward
            
            if side == 'buy':
                return entry_price + reward
            elif side == 'sell':
                return entry_price - reward
            else:
                self.logger.error("거래 방향이 유효하지 않습니다")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"익절가 계산 실패: {str(e)}")
            return 0.0
            
    def calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """
        추세 강도 계산
        
        Args:
            market_data (pd.DataFrame): 시장 데이터
            
        Returns:
            float: 추세 강도 (-1 ~ 1)
        """
        try:
            close = market_data['close']
            
            # 이동평균 계산
            ma20 = self.indicators.sma(close, 20)
            ma50 = self.indicators.sma(close, 50)
            ma200 = self.indicators.sma(close, 200)
            
            # ADX 계산
            adx = self.indicators.adx(
                market_data['high'],
                market_data['low'],
                close,
                14
            )
            
            # 추세 점수 계산
            trend_score = 0.0
            
            # 이동평균 배열 확인
            if ma20.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1]:
                trend_score += 0.3  # 상승 추세
            elif ma20.iloc[-1] < ma50.iloc[-1] < ma200.iloc[-1]:
                trend_score -= 0.3  # 하락 추세
                
            # 단기 추세 확인
            returns = close.pct_change()
            if returns.iloc[-1] > 0:
                trend_score += 0.2
            else:
                trend_score -= 0.2
                
            # ADX 강도 반영
            adx_value = adx['adx'].iloc[-1]
            if not np.isnan(adx_value):
                if adx_value > 25:  # 강한 추세
                    if trend_score > 0:
                        trend_score += 0.3
                    else:
                        trend_score -= 0.3
                        
            # 볼린저 밴드 위치 확인
            bb = self.indicators.bollinger_bands(close, 20, 2.0)
            if close.iloc[-1] > bb['upper'].iloc[-1]:
                trend_score += 0.2
            elif close.iloc[-1] < bb['lower'].iloc[-1]:
                trend_score -= 0.2
                
            return np.clip(trend_score, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"추세 강도 계산 실패: {str(e)}")
            return 0.0
            
    def should_close_position(
        self,
        position: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> bool:
        """
        포지션 종료 여부 확인
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            market_data (pd.DataFrame): 시장 데이터
            
        Returns:
            bool: 종료 여부
        """
        try:
            current_price = market_data['close'].iloc[-1]
            
            # 손절가/익절가 도달 확인
            if position['side'] == 'buy':
                if current_price <= position['stop_loss']:
                    self.logger.info("손절가 도달")
                    return True
                if current_price >= position['take_profit']:
                    self.logger.info("익절가 도달")
                    return True
                    
            elif position['side'] == 'sell':
                if current_price >= position['stop_loss']:
                    self.logger.info("손절가 도달")
                    return True
                if current_price <= position['take_profit']:
                    self.logger.info("익절가 도달")
                    return True
                    
            # 추세 반전 확인
            trend_strength = self.calculate_trend_strength(market_data)
            if position['side'] == 'buy' and trend_strength < -0.5:
                self.logger.info("하락 추세 전환")
                return True
            elif position['side'] == 'sell' and trend_strength > 0.5:
                self.logger.info("상승 추세 전환")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"포지션 종료 여부 확인 실패: {str(e)}")
            return False

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        거래 신호 계산
        
        Args:
            data: OHLCV 데이터
            
        Returns:
            pd.DataFrame: 신호 데이터
        """
        raise NotImplementedError
        
    def execute_trade(self, 
                     data: pd.DataFrame,
                     position: Optional[float] = None,
                     balance: float = 0.0) -> Dict[str, float]:
        """
        거래 실행
        
        Args:
            data: OHLCV 데이터
            position: 현재 포지션
            balance: 현재 잔고
            
        Returns:
            Dict[str, float]: 거래 결과
        """
        raise NotImplementedError

    async def execute(self, *args, **kwargs):
        """전략 실행"""
        raise NotImplementedError("execute 메서드를 구현해야 합니다")
