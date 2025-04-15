import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import talib
from scipy import stats

logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    트레이딩 전략 기본 클래스
    
    주요 기능:
    - 기술적 지표 계산
    - 시그널 생성
    - 포지션 사이징
    - 리스크 관리
    """
    
    def __init__(self,
                 lookback_period: int = 20,
                 rsi_threshold: float = 30.0,
                 volatility_threshold: float = 0.02,
                 min_confidence: float = 0.7):
        """
        전략 초기화
        
        Args:
            lookback_period: 과거 데이터 조회 기간
            rsi_threshold: RSI 과매수/과매도 임계값
            volatility_threshold: 변동성 임계값
            min_confidence: 최소 신뢰도
        """
        self.lookback_period = lookback_period
        self.rsi_threshold = rsi_threshold
        self.volatility_threshold = volatility_threshold
        self.min_confidence = min_confidence
        
        # 데이터 버퍼
        self.price_buffer = []
        self.volume_buffer = []
        self.timestamp_buffer = []
        
    def update_data(self, price: float, volume: float, timestamp: datetime):
        """
        새로운 데이터 업데이트
        
        Args:
            price: 현재 가격
            volume: 현재 거래량
            timestamp: 현재 시간
        """
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        self.timestamp_buffer.append(timestamp)
        
        # 버퍼 크기 유지
        if len(self.price_buffer) > self.lookback_period:
            self.price_buffer.pop(0)
            self.volume_buffer.pop(0)
            self.timestamp_buffer.pop(0)
            
    def calculate_rsi(self) -> float:
        """RSI 계산"""
        if len(self.price_buffer) < self.lookback_period:
            return 50.0
            
        prices = np.array(self.price_buffer)
        rsi = talib.RSI(prices, timeperiod=self.lookback_period)[-1]
        return rsi
        
    def calculate_volatility(self) -> float:
        """변동성 계산"""
        if len(self.price_buffer) < 2:
            return 0.0
            
        returns = np.diff(np.log(self.price_buffer))
        volatility = np.std(returns)
        return volatility
        
    def calculate_momentum(self) -> float:
        """모멘텀 계산"""
        if len(self.price_buffer) < self.lookback_period:
            return 0.0
            
        prices = np.array(self.price_buffer)
        returns = np.diff(np.log(prices))
        momentum = np.sum(returns)
        return momentum
        
    def analyze_market_state(self) -> Dict[str, Any]:
        """
        시장 상태 분석
        
        Returns:
            시장 상태 정보 딕셔너리
        """
        try:
            rsi = self.calculate_rsi()
            volatility = self.calculate_volatility()
            momentum = self.calculate_momentum()
            
            # 시장 상태 판단
            is_overbought = rsi > (100 - self.rsi_threshold)
            is_oversold = rsi < self.rsi_threshold
            is_volatile = volatility > self.volatility_threshold
            is_uptrend = momentum > 0
            is_downtrend = momentum < 0
            
            return {
                'rsi': rsi,
                'volatility': volatility,
                'momentum': momentum,
                'is_overbought': is_overbought,
                'is_oversold': is_oversold,
                'is_volatile': is_volatile,
                'is_uptrend': is_uptrend,
                'is_downtrend': is_downtrend
            }
            
        except Exception as e:
            logger.error(f"시장 상태 분석 중 오류 발생: {e}")
            return {}
            
    def generate_signal(self) -> Tuple[int, float]:
        """
        트레이딩 시그널 생성
        
        Returns:
            (시그널, 신뢰도) 튜플
            시그널: 1(매수), -1(매도), 0(중립)
            신뢰도: 0.0 ~ 1.0
        """
        try:
            market_state = self.analyze_market_state()
            if not market_state:
                return 0, 0.0
                
            # 모멘텀 기반 시그널 생성
            signal = 0
            confidence = 0.0
            
            if market_state['is_uptrend'] and not market_state['is_overbought']:
                signal = 1
                confidence = 0.7
            elif market_state['is_downtrend'] and not market_state['is_oversold']:
                signal = -1
                confidence = 0.7
                
            # 변동성 고려
            if market_state['is_volatile']:
                confidence *= 0.8
                
            # RSI 고려
            if market_state['is_overbought'] or market_state['is_oversold']:
                confidence *= 0.9
                
            return signal, confidence
            
        except Exception as e:
            logger.error(f"시그널 생성 중 오류 발생: {e}")
            return 0, 0.0
            
    def calculate_position_size(self,
                              current_price: float,
                              available_capital: float,
                              risk_per_trade: float = 0.02) -> float:
        """
        포지션 사이즈 계산
        
        Args:
            current_price: 현재 가격
            available_capital: 사용 가능한 자본금
            risk_per_trade: 거래당 리스크 비율
            
        Returns:
            권장 포지션 사이즈
        """
        try:
            # 변동성 기반 포지션 사이징
            volatility = self.calculate_volatility()
            if volatility == 0:
                return 0.0
                
            # 켈리 크라이테리온 적용
            win_rate = 0.6  # 예상 승률
            win_loss_ratio = 1.5  # 예상 승/패 비율
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # 리스크 기반 조정
            position_size = (available_capital * risk_per_trade) / (volatility * current_price)
            position_size *= kelly_fraction
            
            return max(0.0, min(position_size, 1.0))
            
        except Exception as e:
            logger.error(f"포지션 사이즈 계산 중 오류 발생: {e}")
            return 0.0 