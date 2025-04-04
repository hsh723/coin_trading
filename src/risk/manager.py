"""
리스크 관리 모듈
"""

from typing import Dict, Any, List
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger()

class RiskManager:
    """
    리스크 관리자 클래스
    포지션, 손실, 드로다운 등을 관리
    """
    
    def __init__(
        self,
        initial_capital: float,
        risk_per_trade: float,
        max_positions: int,
        daily_loss_limit: float,
        max_drawdown: float
    ):
        """
        리스크 관리자 초기화
        
        Args:
            initial_capital (float): 초기 자본금
            risk_per_trade (float): 거래당 위험 비율
            max_positions (int): 최대 포지션 수
            daily_loss_limit (float): 일일 손실 한도
            max_drawdown (float): 최대 드로다운
        """
        self.logger = logger
        
        # 자본금 관리
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # 리스크 파라미터
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown
        
        # 포지션 관리
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # 손익 관리
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
    def get_capital(self) -> float:
        """
        현재 자본금 조회
        
        Returns:
            float: 현재 자본금
        """
        return self.current_capital
        
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        포지션 목록 조회
        
        Returns:
            List[Dict[str, Any]]: 포지션 목록
        """
        return list(self.positions.values())
        
    def add_position(self, position: Dict[str, Any]) -> bool:
        """
        포지션 추가
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            
        Returns:
            bool: 추가 성공 여부
        """
        try:
            symbol = position['symbol']
            
            if symbol in self.positions:
                self.logger.warning(f"이미 존재하는 포지션: {symbol}")
                return False
                
            if len(self.positions) >= self.max_positions:
                self.logger.warning("최대 포지션 수 초과")
                return False
                
            self.positions[symbol] = position
            self.logger.info(f"포지션 추가: {position}")
            return True
            
        except Exception as e:
            self.logger.error(f"포지션 추가 실패: {str(e)}")
            return False
            
    def remove_position(self, symbol: str) -> bool:
        """
        포지션 제거
        
        Args:
            symbol (str): 심볼
            
        Returns:
            bool: 제거 성공 여부
        """
        try:
            if symbol not in self.positions:
                self.logger.warning(f"존재하지 않는 포지션: {symbol}")
                return False
                
            del self.positions[symbol]
            self.logger.info(f"포지션 제거: {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"포지션 제거 실패: {str(e)}")
            return False
            
    def update_position(
        self,
        symbol: str,
        current_price: float
    ) -> bool:
        """
        포지션 정보 업데이트
        
        Args:
            symbol (str): 심볼
            current_price (float): 현재 가격
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            if symbol not in self.positions:
                return False
                
            position = self.positions[symbol]
            
            # 손익 계산
            pnl = (current_price - position['entry_price']) * position['amount']
            if position['side'] == 'sell':
                pnl = -pnl
                
            # 포지션 정보 업데이트
            position['current_price'] = current_price
            position['unrealized_pnl'] = pnl
            
            return True
            
        except Exception as e:
            self.logger.error(f"포지션 정보 업데이트 실패: {str(e)}")
            return False
            
    def update_daily_pnl(self, pnl: float) -> None:
        """
        일일 손익 업데이트
        
        Args:
            pnl (float): 손익
        """
        try:
            # 일자 변경 확인
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_pnl = 0.0
                self.last_reset_date = current_date
                
            # 손익 업데이트
            self.daily_pnl += pnl
            self.total_pnl += pnl
            
            # 자본금 업데이트
            self.current_capital += pnl
            self.peak_capital = max(self.peak_capital, self.current_capital)
            
            self.logger.info(f"일일 손익 업데이트: {pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"일일 손익 업데이트 실패: {str(e)}")
            
    def check_position_limits(self) -> bool:
        """
        포지션 한도 확인
        
        Returns:
            bool: 한도 내 여부
        """
        return len(self.positions) < self.max_positions
        
    def check_daily_loss_limits(self) -> bool:
        """
        일일 손실 한도 확인
        
        Returns:
            bool: 한도 내 여부
        """
        return self.daily_pnl > -self.initial_capital * self.daily_loss_limit
        
    def check_drawdown_limits(self) -> bool:
        """
        드로다운 한도 확인
        
        Returns:
            bool: 한도 내 여부
        """
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        return drawdown <= self.max_drawdown
        
    def get_position_value(self, symbol: str) -> float:
        """
        포지션 가치 조회
        
        Args:
            symbol (str): 심볼
            
        Returns:
            float: 포지션 가치
        """
        try:
            if symbol not in self.positions:
                return 0.0
                
            position = self.positions[symbol]
            return position['amount'] * position['entry_price']
            
        except Exception as e:
            self.logger.error(f"포지션 가치 조회 실패: {str(e)}")
            return 0.0
            
    def get_total_position_value(self) -> float:
        """
        전체 포지션 가치 조회
        
        Returns:
            float: 전체 포지션 가치
        """
        try:
            total_value = 0.0
            for position in self.positions.values():
                total_value += position['amount'] * position['entry_price']
            return total_value
            
        except Exception as e:
            self.logger.error(f"전체 포지션 가치 조회 실패: {str(e)}")
            return 0.0 