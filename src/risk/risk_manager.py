"""
리스크 관리 시스템 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from ..utils.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RiskMetrics:
    """리스크 지표 데이터 클래스"""
    
    # 포지션 리스크
    position_size: float
    leverage: float
    exposure: float
    margin_level: float
    
    # 손실 리스크
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    potential_loss: float
    max_loss_reached: bool
    
    # 변동성 리스크
    volatility: float
    var_95: float
    expected_shortfall: float
    
    # 집중 리스크
    correlation: float
    concentration: float
    diversification_score: float

class RiskManager:
    """리스크 관리 클래스"""
    
    def __init__(self,
                 initial_capital: float,
                 daily_loss_limit: float = 0.02,
                 weekly_loss_limit: float = 0.05,
                 monthly_loss_limit: float = 0.10,
                 max_drawdown_limit: float = 0.15,
                 max_positions: int = 5,
                 max_position_size: float = 0.10,
                 max_exposure: float = 0.30,
                 volatility_window: int = 20,
                 volatility_threshold: float = 0.02,
                 trailing_stop_activation: float = 0.02,
                 trailing_stop_distance: float = 0.01):
        """
        초기화
        
        Args:
            initial_capital (float): 초기 자본금
            daily_loss_limit (float): 일일 손실 제한 (기본값: 2%)
            weekly_loss_limit (float): 주간 손실 제한 (기본값: 5%)
            monthly_loss_limit (float): 월간 손실 제한 (기본값: 10%)
            max_drawdown_limit (float): 최대 손실 제한 (기본값: 15%)
            max_positions (int): 최대 포지션 수 (기본값: 5)
            max_position_size (float): 최대 포지션 크기 (기본값: 10%)
            max_exposure (float): 최대 노출도 (기본값: 30%)
            volatility_window (int): 변동성 계산 기간 (기본값: 20)
            volatility_threshold (float): 변동성 임계값 (기본값: 2%)
            trailing_stop_activation (float): 트레일링 스탑 활성화 수익률 (기본값: 2%)
            trailing_stop_distance (float): 트레일링 스탑 거리 (기본값: 1%)
        """
        self.logger = get_logger(__name__)
        self.initial_capital = initial_capital
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        self.monthly_loss_limit = monthly_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.max_exposure = max_exposure
        self.volatility_window = volatility_window
        self.volatility_threshold = volatility_threshold
        self.trailing_stop_activation = trailing_stop_activation
        self.trailing_stop_distance = trailing_stop_distance
        
        self._reset_metrics()
        
    def _reset_metrics(self) -> None:
        """리스크 지표 초기화"""
        self.current_capital = self.initial_capital
        self.daily_loss = 0.0
        self.weekly_loss = 0.0
        self.monthly_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = self.initial_capital
        self.positions = {}
        self.trailing_stops = {}
        
    def calculate_position_size(self, price: float, risk_per_trade: float) -> float:
        """
        포지션 크기 계산
        
        Args:
            price (float): 현재 가격
            risk_per_trade (float): 거래당 리스크
            
        Returns:
            float: 포지션 크기
        """
        max_position_value = self.current_capital * self.max_position_size
        position_size = max_position_value / price
        
        # 리스크 기반 포지션 크기 조정
        risk_adjusted_size = (self.current_capital * risk_per_trade) / price
        
        return min(position_size, risk_adjusted_size)
        
    def update_trailing_stop(self, symbol: str, current_price: float) -> Optional[float]:
        """
        트레일링 스탑 업데이트
        
        Args:
            symbol (str): 심볼
            current_price (float): 현재 가격
            
        Returns:
            Optional[float]: 트레일링 스탑 가격
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        entry_price = position['entry_price']
        
        # 트레일링 스탑이 없는 경우 초기화
        if symbol not in self.trailing_stops:
            self.trailing_stops[symbol] = entry_price * (1 - self.trailing_stop_distance)
            
        # 수익이 활성화 수준을 넘으면 트레일링 스탑 업데이트
        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct >= self.trailing_stop_activation:
            new_stop = current_price * (1 - self.trailing_stop_distance)
            self.trailing_stops[symbol] = max(new_stop, self.trailing_stops[symbol])
            
        return self.trailing_stops[symbol]
        
    def check_trading_status(self) -> bool:
        """
        거래 가능 상태 확인
        
        Returns:
            bool: 거래 가능 여부
        """
        # 손실 제한 확인
        if abs(self.daily_loss) >= self.daily_loss_limit * self.initial_capital:
            self.logger.warning("일일 손실 제한 도달")
            return False
            
        if abs(self.weekly_loss) >= self.weekly_loss_limit * self.initial_capital:
            self.logger.warning("주간 손실 제한 도달")
            return False
            
        if abs(self.monthly_loss) >= self.monthly_loss_limit * self.initial_capital:
            self.logger.warning("월간 손실 제한 도달")
            return False
            
        # 최대 손실 확인
        if self.max_drawdown >= self.max_drawdown_limit:
            self.logger.warning("최대 손실 제한 도달")
            return False
            
        # 포지션 수 확인
        if len(self.positions) >= self.max_positions:
            self.logger.warning("최대 포지션 수 도달")
            return False
            
        return True
        
    def update_risk_metrics(self, pnl: float) -> None:
        """
        리스크 지표 업데이트
        
        Args:
            pnl (float): 손익
        """
        self.current_capital += pnl
        
        # 손실 업데이트
        if pnl < 0:
            self.daily_loss += abs(pnl)
            self.weekly_loss += abs(pnl)
            self.monthly_loss += abs(pnl)
            
        # 최대 손실 업데이트
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        else:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        리스크 지표 조회
        
        Returns:
            Dict[str, Any]: 리스크 지표
        """
        return {
            'current_capital': self.current_capital,
            'daily_loss': self.daily_loss,
            'weekly_loss': self.weekly_loss,
            'monthly_loss': self.monthly_loss,
            'max_drawdown': self.max_drawdown,
            'peak_capital': self.peak_capital,
            'positions': len(self.positions),
            'trailing_stops': self.trailing_stops.copy()
        }

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float = 0.95
    ) -> Tuple[float, Dict]:
        """
        적정 포지션 크기 계산
        
        Args:
            entry_price: 진입 가격
            stop_loss: 손절가
            confidence: 신뢰도 (0~1)
            
        Returns:
            Tuple[float, Dict]: (포지션 크기, 리스크 정보)
        """
        try:
            # 리스크 금액 계산
            risk_amount = self.current_capital * self.risk_per_trade
            
            # 손실 비율 계산
            loss_ratio = abs(entry_price - stop_loss) / entry_price
            
            # 기본 포지션 크기 계산
            position_size = risk_amount / (entry_price * loss_ratio)
            
            # 신뢰도에 따른 조정
            position_size *= confidence
            
            # 최대 포지션 크기 제한
            max_size = self.current_capital * self.max_position_size / entry_price
            position_size = min(position_size, max_size)
            
            # 리스크 정보 생성
            risk_info = {
                'position_size': position_size,
                'risk_amount': risk_amount,
                'loss_ratio': loss_ratio,
                'max_loss': risk_amount,
                'confidence': confidence
            }
            
            return position_size, risk_info
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 중 오류 발생: {str(e)}")
            return 0.0, {}
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        volatility: float,
        side: str = 'long',
        atr_multiplier: float = 2.0
    ) -> Tuple[float, float]:
        """
        손절가 및 익절가 계산
        
        Args:
            entry_price: 진입 가격
            volatility: 변동성 (ATR 등)
            side: 포지션 방향 ('long' 또는 'short')
            atr_multiplier: ATR 승수
            
        Returns:
            Tuple[float, float]: (손절가, 익절가)
        """
        try:
            # ATR 기반 손절폭 계산
            stop_distance = volatility * atr_multiplier
            
            if side == 'long':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * 1.5)  # 1.5배 보상비율
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * 1.5)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"손절가 계산 중 오류 발생: {str(e)}")
            return entry_price * 0.95, entry_price * 1.05
    
    def update_risk_metrics(
        self,
        positions: List[Dict],
        market_data: pd.DataFrame
    ) -> RiskMetrics:
        """
        리스크 지표 업데이트
        
        Args:
            positions: 현재 포지션 목록
            market_data: 시장 데이터
            
        Returns:
            RiskMetrics: 리스크 지표
        """
        try:
            # 포지션 리스크 계산
            total_exposure = sum(pos['amount'] * pos['current_price'] for pos in positions)
            leverage = total_exposure / self.current_capital if self.current_capital > 0 else 0
            margin_level = self.current_capital / total_exposure if total_exposure > 0 else 1.0
            
            # 손실 리스크 계산
            unrealized_pnl = sum(pos['unrealized_pnl'] for pos in positions)
            potential_loss = sum(
                pos['amount'] * (pos['current_price'] - pos['stop_loss'])
                for pos in positions if 'stop_loss' in pos
            )
            max_loss_reached = abs(potential_loss) >= self.current_capital * self.max_drawdown
            
            # 변동성 리스크 계산
            if not market_data.empty:
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # 연간화
                var_95 = np.percentile(returns, 5)
                expected_shortfall = returns[returns <= var_95].mean()
            else:
                volatility = 0.0
                var_95 = 0.0
                expected_shortfall = 0.0
            
            # 집중 리스크 계산
            if len(positions) > 1:
                # 상관관계 계산
                symbols = [pos['symbol'] for pos in positions]
                if all(symbol in market_data.columns for symbol in symbols):
                    correlation_matrix = market_data[symbols].corr()
                    correlation = correlation_matrix.mean().mean()
                else:
                    correlation = 0.0
                
                # 집중도 계산
                position_sizes = [pos['amount'] * pos['current_price'] for pos in positions]
                concentration = max(position_sizes) / total_exposure if total_exposure > 0 else 0
                
                # 분산화 점수 계산
                diversification_score = 1 - concentration
            else:
                correlation = 0.0
                concentration = 1.0
                diversification_score = 0.0
            
            return RiskMetrics(
                position_size=total_exposure / self.current_capital,
                leverage=leverage,
                exposure=total_exposure,
                margin_level=margin_level,
                stop_loss_price=0.0,  # 개별 포지션별로 다름
                take_profit_price=0.0,  # 개별 포지션별로 다름
                risk_reward_ratio=1.5,  # 기본값
                potential_loss=potential_loss,
                max_loss_reached=max_loss_reached,
                volatility=volatility,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                correlation=correlation,
                concentration=concentration,
                diversification_score=diversification_score
            )
            
        except Exception as e:
            logger.error(f"리스크 지표 업데이트 중 오류 발생: {str(e)}")
            return None
    
    def check_risk_limits(self, metrics: RiskMetrics) -> List[str]:
        """
        리스크 한도 체크
        
        Args:
            metrics: 리스크 지표
            
        Returns:
            List[str]: 경고 메시지 목록
        """
        warnings = []
        
        # 레버리지 체크
        if metrics.leverage > self.max_leverage:
            warnings.append(f"레버리지 초과: {metrics.leverage:.2f}x > {self.max_leverage:.2f}x")
        
        # 포지션 크기 체크
        if metrics.position_size > self.max_position_size:
            warnings.append(f"포지션 크기 초과: {metrics.position_size:.1%} > {self.max_position_size:.1%}")
        
        # 낙폭 체크
        if metrics.max_loss_reached:
            warnings.append(f"최대 손실 도달: {self.max_drawdown:.1%}")
        
        # 변동성 체크
        if metrics.volatility > self.volatility_threshold:
            warnings.append(f"높은 변동성: {metrics.volatility:.1%} > {self.volatility_threshold:.1%}")
        
        # 상관관계 체크
        if metrics.correlation > self.correlation_threshold:
            warnings.append(f"높은 상관관계: {metrics.correlation:.2f} > {self.correlation_threshold:.2f}")
        
        return warnings
    
    def update_capital(self, pnl: float):
        """
        자본금 업데이트
        
        Args:
            pnl: 손익
        """
        self.current_capital += pnl
        self.daily_pnl.append(pnl)
        
        # 최고점 및 낙폭 업데이트
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
    
    def should_reduce_risk(self) -> Tuple[bool, str]:
        """
        리스크 감소 필요 여부 확인
        
        Returns:
            Tuple[bool, str]: (리스크 감소 필요 여부, 이유)
        """
        # 연속 손실 체크
        recent_trades = self.daily_pnl[-5:]  # 최근 5개 거래
        if len(recent_trades) >= 3 and all(pnl < 0 for pnl in recent_trades[-3:]):
            return True, "연속 손실"
        
        # 낙폭 체크
        if self.current_drawdown >= self.max_drawdown * 0.8:  # 80% 임계값
            return True, f"큰 낙폭 ({self.current_drawdown:.1%})"
        
        # 자본금 감소 체크
        capital_decline = (self.initial_capital - self.current_capital) / self.initial_capital
        if capital_decline >= 0.1:  # 10% 감소
            return True, f"자본금 감소 ({capital_decline:.1%})"
        
        return False, ""
    
    def adjust_position_sizes(self, reduce_factor: float = 0.5) -> Dict[str, float]:
        """
        포지션 크기 조정
        
        Args:
            reduce_factor: 감소 비율
            
        Returns:
            Dict[str, float]: 심볼별 조정된 포지션 크기
        """
        adjusted_sizes = {}
        
        for position in self.positions:
            symbol = position['symbol']
            current_size = position['amount']
            adjusted_size = current_size * reduce_factor
            adjusted_sizes[symbol] = adjusted_size
        
        return adjusted_sizes

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