"""
리스크 관리 모듈
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from ..utils.database import DatabaseManager

class RiskManager:
    """리스크 관리 클래스"""
    
    def __init__(self, db: DatabaseManager):
        """
        초기화
        
        Args:
            db (DatabaseManager): 데이터베이스 관리자
        """
        self.db = db
        self.logger = logging.getLogger(__name__)
        
        # 리스크 파라미터
        self.max_position_size = 0.05  # 초기 자본의 5%
        self.stop_loss = 0.02  # 2%
        self.take_profit = 0.04  # 4%
        self.daily_loss_limit = 0.05  # 5%
        
    async def evaluate_trade(self, trade: Dict[str, Any]) -> bool:
        """
        거래 평가
        
        Args:
            trade (Dict[str, Any]): 거래 정보
            
        Returns:
            bool: 거래 실행 가능 여부
        """
        try:
            # 일일 손실 한도 확인
            if not await self._check_daily_loss_limit():
                return False
                
            # 포지션 크기 확인
            if not await self._check_position_size(trade):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"거래 평가 실패: {str(e)}")
            return False
            
    async def _check_daily_loss_limit(self) -> bool:
        """
        일일 손실 한도 확인
        
        Returns:
            bool: 거래 가능 여부
        """
        try:
            # 오늘의 거래 기록 조회
            today = datetime.now().date()
            trades = await self.db.get_trades_by_date(today)
            
            # 일일 손익 계산
            daily_pnl = sum(trade['pnl'] for trade in trades if 'pnl' in trade)
            
            # 초기 자본 조회
            initial_capital = await self.db.get_initial_capital()
            
            # 일일 손실 한도 확인
            if daily_pnl <= -(initial_capital * self.daily_loss_limit):
                self.logger.warning("일일 손실 한도 도달")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"일일 손실 한도 확인 실패: {str(e)}")
            return False
            
    async def _check_position_size(self, trade: Dict[str, Any]) -> bool:
        """
        포지션 크기 확인
        
        Args:
            trade (Dict[str, Any]): 거래 정보
            
        Returns:
            bool: 포지션 크기 적절 여부
        """
        try:
            # 초기 자본 조회
            initial_capital = await self.db.get_initial_capital()
            
            # 최대 포지션 크기 계산
            max_size = initial_capital * self.max_position_size
            
            # 포지션 크기 확인
            if trade['size'] > max_size:
                self.logger.warning(f"포지션 크기 초과: {trade['size']} > {max_size}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"포지션 크기 확인 실패: {str(e)}")
            return False
            
    async def calculate_stop_loss(self, entry_price: float) -> float:
        """
        손절가 계산
        
        Args:
            entry_price (float): 진입 가격
            
        Returns:
            float: 손절가
        """
        try:
            return entry_price * (1 - self.stop_loss)
            
        except Exception as e:
            self.logger.error(f"손절가 계산 실패: {str(e)}")
            return entry_price * 0.98  # 기본값
            
    async def calculate_take_profit(self, entry_price: float) -> float:
        """
        이익 실현가 계산
        
        Args:
            entry_price (float): 진입 가격
            
        Returns:
            float: 이익 실현가
        """
        try:
            return entry_price * (1 + self.take_profit)
            
        except Exception as e:
            self.logger.error(f"이익 실현가 계산 실패: {str(e)}")
            return entry_price * 1.04  # 기본값
            
    async def update_trade_result(self, pnl: float):
        """
        거래 결과 업데이트
        
        Args:
            pnl (float): 손익
        """
        try:
            # 거래 결과 저장
            await self.db.save_trade_result({
                'timestamp': datetime.now(),
                'pnl': pnl
            })
            
        except Exception as e:
            self.logger.error(f"거래 결과 업데이트 실패: {str(e)}")
            
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """
        리스크 지표 조회
        
        Returns:
            Dict[str, Any]: 리스크 지표
        """
        try:
            # 초기 자본 조회
            initial_capital = await self.db.get_initial_capital()
            
            # 현재 자본 조회
            current_capital = await self.db.get_current_capital()
            
            # 일일 손익 조회
            today = datetime.now().date()
            trades = await self.db.get_trades_by_date(today)
            daily_pnl = sum(trade['pnl'] for trade in trades if 'pnl' in trade)
            
            # 리스크 지표 계산
            return {
                'initial_capital': initial_capital,
                'current_capital': current_capital,
                'daily_pnl': daily_pnl,
                'daily_loss_limit': initial_capital * self.daily_loss_limit,
                'max_position_size': initial_capital * self.max_position_size,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            }
            
        except Exception as e:
            self.logger.error(f"리스크 지표 조회 실패: {str(e)}")
            return {} 