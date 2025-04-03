"""
리스크 관리 모듈

이 모듈은 트레이딩 시스템의 리스크를 관리합니다.
주요 기능:
- 동적 포지션 사이징
- 리스크 평가
- 트레일링 스탑 로스
- 손실 제한
- 시장 변동성 모니터링
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

# 로거 설정
logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    EXTREME = "extreme"

@dataclass
class RiskMetrics:
    volatility: float  # 변동성
    market_condition: MarketCondition  # 시장 상황
    daily_loss: float  # 일일 손실
    weekly_loss: float  # 주간 손실
    monthly_loss: float  # 월간 손실
    max_drawdown: float  # 최대 낙폭
    current_exposure: float  # 현재 노출

@dataclass
class PositionSizingParams:
    base_size: float  # 기본 포지션 크기
    volatility_adjustment: float  # 변동성 조정
    market_condition_adjustment: float  # 시장 상황 조정
    risk_per_trade: float  # 거래당 리스크
    max_position_size: float  # 최대 포지션 크기

class RiskManager:
    """리스크 관리 클래스"""
    
    def __init__(self, config: Dict):
        """
        리스크 관리자 초기화
        
        Args:
            config (Dict): 설정 정보
        """
        self.config = config
        self.initial_capital = config['trading']['initial_capital']
        self.current_capital = self.initial_capital
        
        # 리스크 한도 설정
        self.daily_loss_limit = self.initial_capital * config['trading']['risk']['daily_loss_limit']
        self.weekly_loss_limit = self.initial_capital * config['trading']['risk']['weekly_loss_limit']
        self.monthly_loss_limit = self.initial_capital * config['trading']['risk']['monthly_loss_limit']
        self.max_drawdown_limit = config['trading']['risk']['max_drawdown_limit']
        
        # 손실 추적
        self.daily_loss = 0.0
        self.weekly_loss = 0.0
        self.monthly_loss = 0.0
        self.max_drawdown = 0.0
        
        # 포지션 추적
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        
        # 시장 상황 모니터링
        self.volatility_window = config['trading']['risk']['volatility_window']
        self.volatility_threshold = config['trading']['risk']['volatility_threshold']
        self.market_condition = MarketCondition.NORMAL
        
        # 트레일링 스탑 로스 설정
        self.trailing_stop_activation = config['trading']['risk']['trailing_stop']['activation']
        self.trailing_stop_distance = config['trading']['risk']['trailing_stop']['distance']
        
        logger.info("RiskManager initialized")
    
    def check_trading_status(self, trades: List[Dict], current_data: pd.DataFrame) -> bool:
        """
        거래 가능 여부 확인
        
        Args:
            trades (List[Dict]): 거래 내역
            current_data (pd.DataFrame): 현재 시장 데이터
            
        Returns:
            bool: 거래 가능 여부
        """
        try:
            # 손실 한도 체크
            if not self._check_loss_limits():
                logger.warning("손실 한도 초과로 거래 중단")
                return False
            
            # 시장 상황 체크
            if not self._check_market_conditions(current_data):
                logger.warning("불안정한 시장 상황으로 거래 중단")
                return False
            
            # 포지션 한도 체크
            if not self._check_position_limits():
                logger.warning("포지션 한도 초과로 거래 중단")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"거래 상태 확인 중 오류 발생: {str(e)}")
            return False
    
    def calculate_position_size(
        self,
        price: float,
        volatility: float,
        market_condition: MarketCondition
    ) -> float:
        """
        포지션 크기 계산
        
        Args:
            price (float): 현재 가격
            volatility (float): 변동성
            market_condition (MarketCondition): 시장 상황
            
        Returns:
            float: 포지션 크기
        """
        try:
            # 기본 포지션 크기 계산 (자본의 1%)
            base_size = self.current_capital * 0.01
            
            # 변동성 조정
            volatility_adjustment = 1.0 - (volatility * 2)
            volatility_adjustment = max(0.5, min(1.5, volatility_adjustment))
            
            # 시장 상황 조정
            market_adjustment = {
                MarketCondition.NORMAL: 1.0,
                MarketCondition.VOLATILE: 0.7,
                MarketCondition.EXTREME: 0.5
            }[market_condition]
            
            # 최종 포지션 크기 계산
            position_size = base_size * volatility_adjustment * market_adjustment
            
            # 최대 포지션 크기 제한
            max_size = self.current_capital * self.config['trading']['risk']['max_position_size']
            position_size = min(position_size, max_size)
            
            return position_size
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 중 오류 발생: {str(e)}")
            return 0.0
    
    def update_trailing_stop(
        self,
        position_id: str,
        current_price: float,
        position_type: str
    ) -> Optional[float]:
        """
        트레일링 스탑 로스 업데이트
        
        Args:
            position_id (str): 포지션 ID
            current_price (float): 현재 가격
            position_type (str): 포지션 타입
            
        Returns:
            Optional[float]: 새로운 스탑 로스 가격
        """
        try:
            if position_id not in self.positions:
                return None
            
            position = self.positions[position_id]
            
            # 수익률 계산
            if position_type == 'long':
                profit_rate = (current_price - position['entry_price']) / position['entry_price']
                if profit_rate >= self.trailing_stop_activation:
                    new_stop = current_price * (1 - self.trailing_stop_distance)
                    if new_stop > position.get('stop_loss', 0):
                        position['stop_loss'] = new_stop
                        return new_stop
                        
            elif position_type == 'short':
                profit_rate = (position['entry_price'] - current_price) / position['entry_price']
                if profit_rate >= self.trailing_stop_activation:
                    new_stop = current_price * (1 + self.trailing_stop_distance)
                    if new_stop < position.get('stop_loss', float('inf')):
                        position['stop_loss'] = new_stop
                        return new_stop
            
            return None
            
        except Exception as e:
            logger.error(f"트레일링 스탑 로스 업데이트 중 오류 발생: {str(e)}")
            return None
    
    def update_risk_metrics(
        self,
        trade: Dict,
        current_data: pd.DataFrame
    ) -> RiskMetrics:
        """
        리스크 지표 업데이트
        
        Args:
            trade (Dict): 거래 정보
            current_data (pd.DataFrame): 현재 시장 데이터
            
        Returns:
            RiskMetrics: 업데이트된 리스크 지표
        """
        try:
            # 변동성 계산
            volatility = self._calculate_volatility(current_data)
            
            # 시장 상황 업데이트
            market_condition = self._classify_market_condition(volatility)
            
            # 손실 업데이트
            if trade['pnl'] < 0:
                self.daily_loss += abs(trade['pnl'])
                self.weekly_loss += abs(trade['pnl'])
                self.monthly_loss += abs(trade['pnl'])
            
            # 최대 낙폭 업데이트
            current_drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # 현재 노출 계산
            current_exposure = sum(
                position['size'] * position['entry_price']
                for position in self.positions.values()
            )
            
            return RiskMetrics(
                volatility=volatility,
                market_condition=market_condition,
                daily_loss=self.daily_loss,
                weekly_loss=self.weekly_loss,
                monthly_loss=self.monthly_loss,
                max_drawdown=self.max_drawdown,
                current_exposure=current_exposure
            )
            
        except Exception as e:
            logger.error(f"리스크 지표 업데이트 중 오류 발생: {str(e)}")
            raise
    
    def _check_loss_limits(self) -> bool:
        """
        손실 한도 체크
        
        Returns:
            bool: 손실 한도 준수 여부
        """
        if self.daily_loss > self.daily_loss_limit:
            logger.warning("일일 손실 한도 초과")
            return False
            
        if self.weekly_loss > self.weekly_loss_limit:
            logger.warning("주간 손실 한도 초과")
            return False
            
        if self.monthly_loss > self.monthly_loss_limit:
            logger.warning("월간 손실 한도 초과")
            return False
            
        if self.max_drawdown > self.max_drawdown_limit:
            logger.warning("최대 낙폭 한도 초과")
            return False
            
        return True
    
    def _check_market_conditions(self, current_data: pd.DataFrame) -> bool:
        """
        시장 상황 체크
        
        Args:
            current_data (pd.DataFrame): 현재 시장 데이터
            
        Returns:
            bool: 안정적인 시장 상황 여부
        """
        try:
            # 변동성 계산
            volatility = self._calculate_volatility(current_data)
            
            # 시장 상황 분류
            market_condition = self._classify_market_condition(volatility)
            
            # 극단적인 시장 상황에서는 거래 중단
            if market_condition == MarketCondition.EXTREME:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"시장 상황 체크 중 오류 발생: {str(e)}")
            return False
    
    def _check_position_limits(self) -> bool:
        """
        포지션 한도 체크
        
        Returns:
            bool: 포지션 한도 준수 여부
        """
        # 최대 포지션 수 체크
        if len(self.positions) >= self.config['trading']['risk']['max_positions']:
            return False
            
        # 총 노출 한도 체크
        total_exposure = sum(
            position['size'] * position['entry_price']
            for position in self.positions.values()
        )
        
        if total_exposure > self.current_capital * self.config['trading']['risk']['max_exposure']:
            return False
            
        return True
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        변동성 계산
        
        Args:
            data (pd.DataFrame): 시장 데이터
            
        Returns:
            float: 변동성
        """
        try:
            # 수익률 계산
            returns = data['close'].pct_change()
            
            # 변동성 계산 (표준편차)
            volatility = returns.std() * np.sqrt(252)  # 연간화
            
            return volatility
            
        except Exception as e:
            logger.error(f"변동성 계산 중 오류 발생: {str(e)}")
            return 0.0
    
    def _classify_market_condition(self, volatility: float) -> MarketCondition:
        """
        시장 상황 분류
        
        Args:
            volatility (float): 변동성
            
        Returns:
            MarketCondition: 시장 상황
        """
        if volatility > self.volatility_threshold * 2:
            return MarketCondition.EXTREME
        elif volatility > self.volatility_threshold:
            return MarketCondition.VOLATILE
        else:
            return MarketCondition.NORMAL
    
    def reset_daily_metrics(self):
        """일일 지표 초기화"""
        self.daily_loss = 0.0
    
    def reset_weekly_metrics(self):
        """주간 지표 초기화"""
        self.weekly_loss = 0.0
    
    def reset_monthly_metrics(self):
        """월간 지표 초기화"""
        self.monthly_loss = 0.0 