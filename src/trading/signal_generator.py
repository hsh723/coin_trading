"""
신호 생성 및 통합 모듈
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from ..utils.logger import setup_logger

@dataclass
class Signal:
    type: str  # 'LONG', 'SHORT', 'CLOSE'
    strength: float  # 0.0 ~ 1.0
    price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime
    source: str  # 'TREND', 'FIBONACCI', 'STOCHASTIC', 'TRENDLINE'

@dataclass
class CombinedSignal:
    type: str
    strength: float
    price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime
    trend_signal: Optional[Signal]
    fib_signal: Optional[Signal]
    stoch_signal: Optional[Signal]
    trendline_signal: Optional[Signal]

class SignalGenerator:
    """
    거래 신호 생성 및 통합 클래스
    """
    
    def __init__(self):
        """
        신호 생성기 초기화
        """
        # 로거 설정
        self.logger = setup_logger()
        
        # 신호 우선순위 가중치
        self.signal_weights = {
            'TREND': 0.4,
            'FIBONACCI': 0.3,
            'STOCHASTIC': 0.2,
            'TRENDLINE': 0.1
        }
        
        # 최소 신호 강도
        self.min_signal_strength = 0.6
        
        # 신호 충돌 해결 규칙
        self.conflict_rules = {
            ('LONG', 'SHORT'): 'CLOSE',
            ('SHORT', 'LONG'): 'CLOSE',
            ('CLOSE', 'LONG'): 'CLOSE',
            ('CLOSE', 'SHORT'): 'CLOSE'
        }
        
        self.logger.info("SignalGenerator initialized")
    
    def combine_signals(
        self,
        trend_signal: Optional[Signal],
        fib_signal: Optional[Signal],
        stoch_signal: Optional[Signal],
        trendline_signal: Optional[Signal]
    ) -> Optional[CombinedSignal]:
        """
        개별 신호들을 통합하여 최종 신호 생성
        
        Args:
            trend_signal (Optional[Signal]): 트렌드 신호
            fib_signal (Optional[Signal]): 피보나치 신호
            stoch_signal (Optional[Signal]): 스토캐스틱 신호
            trendline_signal (Optional[Signal]): 추세선 신호
            
        Returns:
            Optional[CombinedSignal]: 통합된 신호
        """
        try:
            # 신호 필터링
            signals = self.filter_signals([
                trend_signal,
                fib_signal,
                stoch_signal,
                trendline_signal
            ])
            
            if not signals:
                return None
                
            # 신호 강도 계산
            strength = self.calculate_signal_strength(signals)
            
            if strength < self.min_signal_strength:
                return None
                
            # 최종 결정
            final_type = self.get_final_decision(signals)
            
            # 신호 충돌 확인
            if self._check_signal_conflicts(signals):
                final_type = 'CLOSE'
            
            # 진입/청산 가격 결정
            price = self._determine_entry_price(signals, final_type)
            stop_loss, take_profit = self._determine_stop_levels(signals, final_type, price)
            
            # 신호 사유 생성
            reason = self._generate_signal_reason(signals, final_type)
            
            # 통합 신호 생성
            combined_signal = CombinedSignal(
                type=final_type,
                strength=strength,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
                timestamp=datetime.now(),
                trend_signal=trend_signal,
                fib_signal=fib_signal,
                stoch_signal=stoch_signal,
                trendline_signal=trendline_signal
            )
            
            self.logger.info(
                f"Combined signal generated: {combined_signal.type} "
                f"(strength: {combined_signal.strength:.2f})"
            )
            
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {str(e)}")
            raise
    
    def calculate_signal_strength(
        self,
        signals: List[Signal]
    ) -> float:
        """
        신호 강도 계산
        
        Args:
            signals (List[Signal]): 신호 목록
            
        Returns:
            float: 신호 강도 (0.0 ~ 1.0)
        """
        if not signals:
            return 0.0
            
        # 각 신호의 가중치 적용
        weighted_strengths = []
        for signal in signals:
            weight = self.signal_weights.get(signal.source, 0.0)
            weighted_strengths.append(signal.strength * weight)
            
        # 가중 평균 계산
        total_strength = sum(weighted_strengths)
        total_weight = sum(self.signal_weights.get(signal.source, 0.0) for signal in signals)
        
        if total_weight == 0:
            return 0.0
            
        return total_strength / total_weight
    
    def filter_signals(
        self,
        signals: List[Optional[Signal]],
        min_strength: float = 0.6
    ) -> List[Signal]:
        """
        약한 신호 필터링
        
        Args:
            signals (List[Optional[Signal]]): 신호 목록
            min_strength (float): 최소 신호 강도
            
        Returns:
            List[Signal]: 필터링된 신호 목록
        """
        return [
            signal for signal in signals
            if signal is not None and signal.strength >= min_strength
        ]
    
    def get_final_decision(
        self,
        signals: List[Signal]
    ) -> str:
        """
        최종 진입/청산 결정
        
        Args:
            signals (List[Signal]): 신호 목록
            
        Returns:
            str: 최종 결정 ('LONG', 'SHORT', 'CLOSE')
        """
        if not signals:
            return 'CLOSE'
            
        # 트렌드 신호 확인
        trend_signal = next(
            (s for s in signals if s.source == 'TREND'),
            None
        )
        
        if not trend_signal:
            return 'CLOSE'
            
        # 트렌드 방향에 따른 신호 필터링
        valid_signals = [
            s for s in signals
            if s.type == trend_signal.type
        ]
        
        if not valid_signals:
            return 'CLOSE'
            
        # 신호 강도 기반 결정
        strengths = [s.strength for s in valid_signals]
        max_strength = max(strengths)
        
        if max_strength < self.min_signal_strength:
            return 'CLOSE'
            
        return trend_signal.type
    
    def _check_signal_conflicts(
        self,
        signals: List[Signal]
    ) -> bool:
        """
        신호 충돌 확인
        
        Args:
            signals (List[Signal]): 신호 목록
            
        Returns:
            bool: 충돌 여부
        """
        if len(signals) < 2:
            return False
            
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                signal1, signal2 = signals[i], signals[j]
                if (signal1.type, signal2.type) in self.conflict_rules:
                    return True
                    
        return False
    
    def _determine_entry_price(
        self,
        signals: List[Signal],
        signal_type: str
    ) -> float:
        """
        진입 가격 결정
        
        Args:
            signals (List[Signal]): 신호 목록
            signal_type (str): 신호 타입
            
        Returns:
            float: 진입 가격
        """
        # 피보나치 신호 우선
        fib_signal = next(
            (s for s in signals if s.source == 'FIBONACCI'),
            None
        )
        
        if fib_signal and fib_signal.type == signal_type:
            return fib_signal.price
            
        # 다른 신호들의 평균 가격
        valid_signals = [s for s in signals if s.type == signal_type]
        if valid_signals:
            return np.mean([s.price for s in valid_signals])
            
        return 0.0
    
    def _determine_stop_levels(
        self,
        signals: List[Signal],
        signal_type: str,
        entry_price: float
    ) -> Tuple[float, float]:
        """
        손절/익절 가격 결정
        
        Args:
            signals (List[Signal]): 신호 목록
            signal_type (str): 신호 타입
            entry_price (float): 진입 가격
            
        Returns:
            Tuple[float, float]: (손절가, 익절가)
        """
        # 각 신호의 손절/익절 가격 평균
        stop_losses = []
        take_profits = []
        
        for signal in signals:
            if signal.type == signal_type:
                stop_losses.append(signal.stop_loss)
                take_profits.append(signal.take_profit)
                
        if not stop_losses or not take_profits:
            # 기본 손절/익절 설정
            if signal_type == 'LONG':
                return entry_price * 0.98, entry_price * 1.02
            else:
                return entry_price * 1.02, entry_price * 0.98
                
        return np.mean(stop_losses), np.mean(take_profits)
    
    def _generate_signal_reason(
        self,
        signals: List[Signal],
        signal_type: str
    ) -> str:
        """
        신호 사유 생성
        
        Args:
            signals (List[Signal]): 신호 목록
            signal_type (str): 신호 타입
            
        Returns:
            str: 신호 사유
        """
        reasons = []
        
        for signal in signals:
            if signal.type == signal_type:
                reasons.append(f"{signal.source}: {signal.reason}")
                
        return " | ".join(reasons) 