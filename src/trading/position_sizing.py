import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from ..utils.logger import setup_logger

@dataclass
class PositionSize:
    size: float  # 포지션 크기
    leverage: float  # 레버리지
    risk_amount: float  # 위험 금액
    entry_price: float  # 진입 가격
    stop_loss: float  # 손절 가격
    take_profit: float  # 익절 가격
    risk_reward_ratio: float  # 손익비
    volatility_adjustment: float  # 변동성 조정 계수

@dataclass
class EntryScaling:
    entries: List[Dict[str, float]]  # 진입 정보 리스트
    total_risk: float  # 총 위험 금액
    total_position: float  # 총 포지션 크기
    average_entry: float  # 평균 진입 가격

class PositionSizing:
    def __init__(self):
        self.logger = setup_logger()
        self.max_risk_per_trade = 0.02  # 최대 거래당 위험 (2%)
        self.max_total_risk = 0.06  # 최대 총 위험 (6%)
        self.base_risk_reward = 1.0  # 기본 손익비
        self.volatility_threshold = 0.03  # 변동성 임계값 (3%)
        self.max_leverage = 10.0  # 최대 레버리지
        self.min_leverage = 1.0  # 최소 레버리지

    def calculate_position_size(
        self,
        balance: float,
        risk_percentage: float,
        entry: float,
        stop_loss: float,
        leverage: float = 1.0
    ) -> PositionSize:
        """
        포지션 크기 계산
        
        Args:
            balance (float): 계좌 잔고
            risk_percentage (float): 위험 비율 (0.0 ~ 1.0)
            entry (float): 진입 가격
            stop_loss (float): 손절 가격
            leverage (float): 레버리지
            
        Returns:
            PositionSize: 포지션 크기 정보
        """
        # 레버리지 제한
        leverage = max(min(leverage, self.max_leverage), self.min_leverage)
        
        # 위험 금액 계산
        risk_amount = balance * risk_percentage
        
        # 가격 변동폭 계산
        price_range = abs(entry - stop_loss)
        
        # 기본 포지션 크기 계산
        base_size = risk_amount / price_range
        
        # 레버리지 적용
        leveraged_size = base_size * leverage
        
        # 손익비 계산
        risk_reward_ratio = self.calculate_risk_reward_ratio(entry, stop_loss)
        
        # 변동성 조정
        volatility = self._calculate_volatility(entry, stop_loss)
        adjusted_size = self.adjust_position_for_volatility(leveraged_size, volatility)
        
        # 익절 가격 계산
        take_profit = self._calculate_take_profit(entry, stop_loss, risk_reward_ratio)
        
        return PositionSize(
            size=adjusted_size,
            leverage=leverage,
            risk_amount=risk_amount,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            volatility_adjustment=volatility
        )

    def calculate_entry_scaling(
        self,
        balance: float,
        entry_count: int,
        entry: float,
        stop_loss: float,
        leverage: float = 1.0
    ) -> EntryScaling:
        """
        단계별 진입 비중 계산
        
        Args:
            balance (float): 계좌 잔고
            entry_count (int): 진입 횟수
            entry (float): 진입 가격
            stop_loss (float): 손절 가격
            leverage (float): 레버리지
            
        Returns:
            EntryScaling: 진입 스케일링 정보
        """
        # 진입 비중 설정
        entry_weights = [0.07, 0.13, 0.40]  # 7%, 13%, 40%
        
        # 진입 정보 리스트
        entries = []
        total_risk = 0.0
        total_position = 0.0
        weighted_entry = 0.0
        
        for i in range(min(entry_count, len(entry_weights))):
            # 각 진입의 위험 비율 계산
            risk_percentage = entry_weights[i] * self.max_risk_per_trade
            
            # 포지션 크기 계산
            position = self.calculate_position_size(
                balance=balance,
                risk_percentage=risk_percentage,
                entry=entry,
                stop_loss=stop_loss,
                leverage=leverage
            )
            
            # 진입 정보 저장
            entries.append({
                'weight': entry_weights[i],
                'risk_percentage': risk_percentage,
                'size': position.size,
                'risk_amount': position.risk_amount
            })
            
            # 총계 업데이트
            total_risk += position.risk_amount
            total_position += position.size
            weighted_entry += entry * entry_weights[i]
            
        return EntryScaling(
            entries=entries,
            total_risk=total_risk,
            total_position=total_position,
            average_entry=weighted_entry
        )

    def calculate_risk_reward_ratio(
        self,
        entry: float,
        take_profit: float,
        stop_loss: float
    ) -> float:
        """
        손익비 계산
        
        Args:
            entry (float): 진입 가격
            take_profit (float): 익절 가격
            stop_loss (float): 손절 가격
            
        Returns:
            float: 손익비
        """
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return 0.0
            
        return reward / risk

    def adjust_position_for_volatility(
        self,
        size: float,
        volatility: float
    ) -> float:
        """
        변동성 기반 포지션 조정
        
        Args:
            size (float): 원래 포지션 크기
            volatility (float): 변동성
            
        Returns:
            float: 조정된 포지션 크기
        """
        if volatility <= self.volatility_threshold:
            return size
            
        # 변동성이 높을수록 포지션 크기 감소
        adjustment_factor = 1 - (volatility - self.volatility_threshold)
        adjustment_factor = max(0.5, min(1.0, adjustment_factor))
        
        return size * adjustment_factor

    def _calculate_volatility(self, entry: float, stop_loss: float) -> float:
        """변동성 계산"""
        return abs(entry - stop_loss) / entry

    def _calculate_take_profit(
        self,
        entry: float,
        stop_loss: float,
        risk_reward_ratio: float
    ) -> float:
        """익절 가격 계산"""
        risk = abs(entry - stop_loss)
        reward = risk * risk_reward_ratio
        
        if entry > stop_loss:
            return entry + reward
        else:
            return entry - reward

    def validate_position(
        self,
        position: PositionSize,
        total_risk: float
    ) -> bool:
        """
        포지션 유효성 검증
        
        Args:
            position (PositionSize): 포지션 정보
            total_risk (float): 현재 총 위험 금액
            
        Returns:
            bool: 유효성 여부
        """
        # 레버리지 검증
        if position.leverage > self.max_leverage:
            self.logger.warning(f"레버리지 초과: {position.leverage}")
            return False
            
        # 위험 한도 검증
        if position.risk_amount / position.entry_price > self.max_risk_per_trade:
            self.logger.warning(f"거래당 위험 초과: {position.risk_amount / position.entry_price}")
            return False
            
        # 총 위험 한도 검증
        if total_risk + position.risk_amount > self.max_total_risk:
            self.logger.warning(f"총 위험 한도 초과: {total_risk + position.risk_amount}")
            return False
            
        # 손익비 검증
        if position.risk_reward_ratio < self.base_risk_reward:
            self.logger.warning(f"손익비 미달: {position.risk_reward_ratio}")
            return False
            
        return True 