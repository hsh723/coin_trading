from typing import Dict, Any, Optional
from ..utils.logger import setup_logger

class RiskManager:
    """리스크 관리자 클래스"""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        리스크 관리자 초기화
        
        Args:
            initial_capital (float): 초기 자본금
        """
        self.logger = setup_logger('risk_manager')
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = 0.02  # 거래당 리스크 2%
        self.max_daily_loss = 0.05  # 일일 최대 손실 5%
        self.max_position_size = 0.1  # 최대 포지션 크기 10%
        self.daily_loss = 0.0
        self.trades_today = 0
        self.max_trades_per_day = 10
        
    def update_capital(self, new_capital: float) -> None:
        """
        자본금 업데이트
        
        Args:
            new_capital (float): 새로운 자본금
        """
        self.current_capital = new_capital
        self.logger.info(f"자본금 업데이트: {new_capital:.2f}")
        
    def can_trade(self) -> bool:
        """
        거래 가능 여부 확인
        
        Returns:
            bool: 거래 가능 여부
        """
        try:
            # 일일 최대 손실 체크
            if self.daily_loss >= self.max_daily_loss * self.initial_capital:
                self.logger.warning("일일 최대 손실 한도 도달")
                return False
                
            # 일일 최대 거래 횟수 체크
            if self.trades_today >= self.max_trades_per_day:
                self.logger.warning("일일 최대 거래 횟수 도달")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"거래 가능 여부 확인 실패: {str(e)}")
            return False
            
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        포지션 크기 계산
        
        Args:
            entry_price (float): 진입 가격
            stop_loss (float): 손절 가격
            
        Returns:
            float: 포지션 크기
        """
        try:
            # 리스크 금액 계산
            risk_amount = self.current_capital * self.risk_per_trade
            
            # 손절폭 계산
            stop_loss_pips = abs(entry_price - stop_loss)
            
            # 포지션 크기 계산
            position_size = risk_amount / stop_loss_pips
            
            # 최대 포지션 크기 제한
            max_position = self.current_capital * self.max_position_size
            position_size = min(position_size, max_position)
            
            self.logger.info(f"포지션 크기 계산: {position_size:.2f}")
            return position_size
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {str(e)}")
            return 0.0
            
    def update_trade_result(self, pnl: float) -> None:
        """
        거래 결과 업데이트
        
        Args:
            pnl (float): 손익
        """
        try:
            # 자본금 업데이트
            self.current_capital += pnl
            
            # 일일 손실 업데이트
            if pnl < 0:
                self.daily_loss += abs(pnl)
                
            # 거래 횟수 업데이트
            self.trades_today += 1
            
            self.logger.info(f"거래 결과 업데이트: PNL={pnl:.2f}, 자본금={self.current_capital:.2f}")
            
        except Exception as e:
            self.logger.error(f"거래 결과 업데이트 실패: {str(e)}")
            
    def reset_daily_stats(self) -> None:
        """일일 통계 초기화"""
        try:
            self.daily_loss = 0.0
            self.trades_today = 0
            self.logger.info("일일 통계 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"일일 통계 초기화 실패: {str(e)}")
            
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        리스크 메트릭 조회
        
        Returns:
            Dict[str, Any]: 리스크 메트릭
        """
        try:
            return {
                'current_capital': self.current_capital,
                'daily_loss': self.daily_loss,
                'max_daily_loss': self.max_daily_loss * self.initial_capital,
                'trades_today': self.trades_today,
                'max_trades_per_day': self.max_trades_per_day,
                'risk_per_trade': self.risk_per_trade,
                'max_position_size': self.max_position_size
            }
            
        except Exception as e:
            self.logger.error(f"리스크 메트릭 조회 실패: {str(e)}")
            return {}
            
    def adjust_risk_parameters(self, market_volatility: float) -> None:
        """
        리스크 파라미터 조정
        
        Args:
            market_volatility (float): 시장 변동성
        """
        try:
            # 변동성에 따른 리스크 조정
            if market_volatility > 0.05:  # 변동성 높음
                self.risk_per_trade = 0.01  # 리스크 감소
                self.max_position_size = 0.05  # 포지션 크기 감소
            elif market_volatility < 0.02:  # 변동성 낮음
                self.risk_per_trade = 0.03  # 리스크 증가
                self.max_position_size = 0.15  # 포지션 크기 증가
            else:  # 변동성 보통
                self.risk_per_trade = 0.02
                self.max_position_size = 0.1
                
            self.logger.info(f"리스크 파라미터 조정: 변동성={market_volatility:.4f}")
            
        except Exception as e:
            self.logger.error(f"리스크 파라미터 조정 실패: {str(e)}") 