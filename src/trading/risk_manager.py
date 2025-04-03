"""
리스크 관리 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger()

class RiskManager:
    def __init__(
        self,
        config: Dict[str, Any],
        initial_capital: float
    ):
        """
        리스크 관리자 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
            initial_capital (float): 초기 자본금
        """
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.logger = setup_logger()
        
        # 리스크 관리 설정
        self.risk_per_trade = config.get('risk_management', {}).get('risk_per_trade', 0.02)  # 거래당 리스크
        self.max_daily_loss = config.get('risk_management', {}).get('max_daily_loss', 0.05)  # 일일 최대 손실
        self.max_drawdown = config.get('risk_management', {}).get('max_drawdown', 0.15)  # 최대 낙폭
        self.max_positions = config.get('risk_management', {}).get('max_positions', 5)  # 최대 포지션 수
        self.max_position_size = config.get('risk_management', {}).get('max_position_size', 0.2)  # 최대 포지션 크기
        
        # 상태 변수
        self.daily_pnl = 0.0
        self.peak_capital = initial_capital
        self.open_positions: List[Dict[str, Any]] = []
        self.daily_trades: List[Dict[str, Any]] = []
        
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        symbol: str
    ) -> float:
        """
        포지션 크기 계산
        
        Args:
            entry_price (float): 진입 가격
            stop_loss (float): 손절 가격
            symbol (str): 거래 심볼
            
        Returns:
            float: 포지션 크기 (수량)
        """
        try:
            # 리스크 금액 계산
            risk_amount = self.current_capital * self.risk_per_trade
            
            # 가격 변동폭 계산
            price_diff = abs(entry_price - stop_loss)
            
            # 포지션 크기 계산
            position_size = risk_amount / price_diff
            
            # 최대 포지션 크기 제한
            max_size = self.current_capital * self.max_position_size / entry_price
            position_size = min(position_size, max_size)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {str(e)}")
            return 0.0
            
    def check_daily_loss_limit(self) -> bool:
        """
        일일 손실 한도 확인
        
        Returns:
            bool: 거래 가능 여부
        """
        try:
            # 일일 손실 한도 확인
            if self.daily_pnl <= -self.initial_capital * self.max_daily_loss:
                self.logger.warning("일일 손실 한도 도달")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"일일 손실 한도 확인 실패: {str(e)}")
            return False
            
    def check_drawdown_limit(self) -> bool:
        """
        낙폭 한도 확인
        
        Returns:
            bool: 거래 가능 여부
        """
        try:
            # 최대 낙폭 확인
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            
            if current_drawdown >= self.max_drawdown:
                self.logger.warning("최대 낙폭 한도 도달")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"낙폭 한도 확인 실패: {str(e)}")
            return False
            
    def check_position_limit(self) -> bool:
        """
        포지션 한도 확인
        
        Returns:
            bool: 거래 가능 여부
        """
        try:
            # 최대 포지션 수 확인
            if len(self.open_positions) >= self.max_positions:
                self.logger.warning("최대 포지션 수 도달")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"포지션 한도 확인 실패: {str(e)}")
            return False
            
    def update_position(
        self,
        position: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """
        포지션 정보 업데이트
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            current_price (float): 현재 가격
            
        Returns:
            Dict[str, Any]: 업데이트된 포지션 정보
        """
        try:
            # 손익 계산
            if position['side'] == 'buy':
                pnl = (current_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - current_price) * position['size']
                
            # 포지션 정보 업데이트
            position['current_price'] = current_price
            position['unrealized_pnl'] = pnl
            position['total_pnl'] = pnl
            
            return position
            
        except Exception as e:
            self.logger.error(f"포지션 정보 업데이트 실패: {str(e)}")
            return position
            
    def add_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: float
    ) -> bool:
        """
        포지션 추가
        
        Args:
            symbol (str): 거래 심볼
            side (str): 포지션 방향
            entry_price (float): 진입 가격
            size (float): 포지션 크기
            stop_loss (float): 손절 가격
            take_profit (float): 익절 가격
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 포지션 한도 확인
            if not self.check_position_limit():
                return False
                
            # 포지션 정보 생성
            position = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'size': size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(),
                'current_price': entry_price,
                'unrealized_pnl': 0.0,
                'total_pnl': 0.0
            }
            
            # 포지션 추가
            self.open_positions.append(position)
            
            return True
            
        except Exception as e:
            self.logger.error(f"포지션 추가 실패: {str(e)}")
            return False
            
    def remove_position(
        self,
        position: Dict[str, Any],
        exit_price: float,
        reason: str
    ) -> bool:
        """
        포지션 제거
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            exit_price (float): 청산 가격
            reason (str): 청산 사유
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 손익 계산
            if position['side'] == 'buy':
                pnl = (exit_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - exit_price) * position['size']
                
            # 포지션 정보 업데이트
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['realized_pnl'] = pnl
            position['total_pnl'] = pnl
            position['exit_reason'] = reason
            
            # 거래 기록 추가
            self.daily_trades.append(position)
            
            # 일일 손익 업데이트
            self.daily_pnl += pnl
            
            # 자본금 업데이트
            self.current_capital += pnl
            
            # 최대 자본금 업데이트
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
                
            # 포지션 제거
            self.open_positions.remove(position)
            
            return True
            
        except Exception as e:
            self.logger.error(f"포지션 제거 실패: {str(e)}")
            return False
            
    def check_stop_loss(
        self,
        position: Dict[str, Any],
        current_price: float
    ) -> bool:
        """
        손절 조건 확인
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            current_price (float): 현재 가격
            
        Returns:
            bool: 손절 필요 여부
        """
        try:
            if position['side'] == 'buy':
                return current_price <= position['stop_loss']
            else:
                return current_price >= position['stop_loss']
                
        except Exception as e:
            self.logger.error(f"손절 조건 확인 실패: {str(e)}")
            return False
            
    def check_take_profit(
        self,
        position: Dict[str, Any],
        current_price: float
    ) -> bool:
        """
        익절 조건 확인
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            current_price (float): 현재 가격
            
        Returns:
            bool: 익절 필요 여부
        """
        try:
            if position['side'] == 'buy':
                return current_price >= position['take_profit']
            else:
                return current_price <= position['take_profit']
                
        except Exception as e:
            self.logger.error(f"익절 조건 확인 실패: {str(e)}")
            return False
            
    def reset_daily_stats(self):
        """
        일일 통계 초기화
        """
        try:
            self.daily_pnl = 0.0
            self.daily_trades = []
            
        except Exception as e:
            self.logger.error(f"일일 통계 초기화 실패: {str(e)}")
            
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        리스크 지표 조회
        
        Returns:
            Dict[str, float]: 리스크 지표
        """
        try:
            return {
                'current_capital': self.current_capital,
                'peak_capital': self.peak_capital,
                'daily_pnl': self.daily_pnl,
                'drawdown': (self.peak_capital - self.current_capital) / self.peak_capital,
                'open_positions': len(self.open_positions),
                'daily_trades': len(self.daily_trades)
            }
            
        except Exception as e:
            self.logger.error(f"리스크 지표 조회 실패: {str(e)}")
            return {} 