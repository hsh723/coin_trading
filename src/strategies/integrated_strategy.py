from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from .base import BaseStrategy, TrendType, TrendInfo, FibonacciLevels, StochasticSignal, TrendlineInfo

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class Signal:
    type: SignalType
    strength: float  # 0.0 ~ 1.0
    price: float
    stop_loss: float
    take_profit: float
    reason: str

class IntegratedStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.signal_weights = config.get('signal_weights', {
            'trend': 0.3,
            'fibonacci': 0.2,
            'stochastic': 0.2,
            'trendline': 0.3
        })
        self.min_signal_strength = config.get('min_signal_strength', 0.6)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('take_profit_pct', 0.04)

    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """통합 신호 생성"""
        # 각 지표별 신호 분석
        trend_info = self.identify_trend(data)
        fib_levels = self.calculate_fibonacci_levels(data)
        stoch_signal = self.check_stochastic(data)
        trendline_info = self.detect_trendlines(data)
        
        # 각 지표별 신호 점수 계산
        trend_score = self._calculate_trend_score(trend_info)
        fib_score = self._calculate_fibonacci_score(fib_levels)
        stoch_score = self._calculate_stochastic_score(stoch_signal)
        trendline_score = self._calculate_trendline_score(trendline_info)
        
        # 종합 점수 계산
        total_score = (
            trend_score * self.signal_weights['trend'] +
            fib_score * self.signal_weights['fibonacci'] +
            stoch_score * self.signal_weights['stochastic'] +
            trendline_score * self.signal_weights['trendline']
        )
        
        # 신호 생성
        signal = self._generate_signal(total_score, data)
        
        return {
            'signal': signal,
            'scores': {
                'trend': trend_score,
                'fibonacci': fib_score,
                'stochastic': stoch_score,
                'trendline': trendline_score,
                'total': total_score
            },
            'indicators': {
                'trend': trend_info,
                'fibonacci': fib_levels,
                'stochastic': stoch_signal,
                'trendline': trendline_info
            }
        }

    def _calculate_trend_score(self, trend_info: TrendInfo) -> float:
        """트렌드 점수 계산"""
        score = 0.0
        
        # 트렌드 타입에 따른 점수
        if trend_info.trend_type == TrendType.UPTREND:
            score += 0.4
        elif trend_info.trend_type == TrendType.DOWNTREND:
            score -= 0.4
            
        # MA 크로스오버 점수
        if trend_info.ma_crossover:
            score += 0.3
            
        # 트렌드 강도 점수
        score += trend_info.strength * 0.3
        
        return max(min(score, 1.0), -1.0)

    def _calculate_fibonacci_score(self, fib_levels: FibonacciLevels) -> float:
        """피보나치 점수 계산"""
        score = 0.0
        
        # 현재 가격이 피보나치 레벨에 가까울수록 점수 증가
        distance = fib_levels.distance_to_level
        if distance < 0.1:
            score += 0.5
            
        # 레벨에 따른 점수 조정
        level = float(fib_levels.nearest_level)
        if level in [0.382, 0.618]:
            score += 0.3
        elif level in [0.5]:
            score += 0.2
            
        return max(min(score, 1.0), -1.0)

    def _calculate_stochastic_score(self, stoch_signal: StochasticSignal) -> float:
        """스토캐스틱 점수 계산"""
        score = 0.0
        
        # 과매수/과매도 점수
        if stoch_signal.is_overbought:
            score -= 0.4
        elif stoch_signal.is_oversold:
            score += 0.4
            
        # 크로스오버 점수
        if stoch_signal.crossover_up:
            score += 0.3
        elif stoch_signal.crossover_down:
            score -= 0.3
            
        return max(min(score, 1.0), -1.0)

    def _calculate_trendline_score(self, trendline_info: TrendlineInfo) -> float:
        """추세선 점수 계산"""
        score = 0.0
        
        # R-squared 값에 따른 점수
        score += trendline_info.r_squared * 0.4
        
        # 기울기에 따른 점수
        if trendline_info.slope > 0:
            score += 0.3
        else:
            score -= 0.3
            
        return max(min(score, 1.0), -1.0)

    def _generate_signal(self, total_score: float, data: pd.DataFrame) -> Signal:
        """최종 신호 생성"""
        current_price = data['close'].iloc[-1]
        
        # 신호 타입 결정
        if total_score > 0.7:
            signal_type = SignalType.STRONG_BUY
        elif total_score > 0.3:
            signal_type = SignalType.BUY
        elif total_score < -0.7:
            signal_type = SignalType.STRONG_SELL
        elif total_score < -0.3:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
            
        # 신호 강도 계산
        strength = abs(total_score)
        
        # 손절/익절 가격 계산
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
        else:
            stop_loss = take_profit = current_price
            
        # 신호 사유 생성
        reason = self._generate_signal_reason(signal_type, total_score)
        
        return Signal(
            type=signal_type,
            strength=strength,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason
        )

    def _generate_signal_reason(self, signal_type: SignalType, total_score: float) -> str:
        """신호 사유 생성"""
        reasons = {
            SignalType.STRONG_BUY: "강한 매수 신호: 모든 지표가 상승 추세를 지지",
            SignalType.BUY: "매수 신호: 대부분의 지표가 상승 추세를 지지",
            SignalType.NEUTRAL: "중립 신호: 지표들이 혼조세를 보임",
            SignalType.SELL: "매도 신호: 대부분의 지표가 하락 추세를 지지",
            SignalType.STRONG_SELL: "강한 매도 신호: 모든 지표가 하락 추세를 지지"
        }
        return reasons.get(signal_type, "알 수 없는 신호") 