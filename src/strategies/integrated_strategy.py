from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from scipy import stats

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
        
        # 이동평균선 설정
        self.ma_short = 20  # 20일 이동평균
        self.ma_long = 60   # 60일 이동평균
        self.ma_200 = 200   # 200일 이동평균
        
        # 볼린저 밴드 설정
        self.bb_period = 20
        self.bb_std = 2
        
        # RSI 설정
        self.rsi_period = 14
        
        # 스토캐스틱 설정
        self.stoch_period = 14
        
        # 리스크 관리
        self.risk_per_trade = 0.1  # 전체 자본의 10% 리스크
        self.take_profit_ratio = 0.15  # 15% 익절
        self.stop_loss_ratio = 0.15  # 15% 손절
        
        # 추세선 설정
        self.touch_threshold = 0.002  # 0.2% 이내면 터치로 판단
        self.min_touch_points = 3  # 최소 터치 포인트 수
        self.trendline_lookback = 50  # 추세선을 찾을 기간

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 기술적 지표를 계산합니다."""
        # 이동평균선
        df['ma_short'] = SMAIndicator(close=df['close'], window=self.ma_short).sma_indicator()
        df['ma_long'] = SMAIndicator(close=df['close'], window=self.ma_long).sma_indicator()
        df['ma_200'] = SMAIndicator(close=df['close'], window=self.ma_200).sma_indicator()
        
        # 볼린저 밴드
        bollinger = BollingerBands(close=df['close'], window=self.bb_period, window_dev=self.bb_std)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=self.rsi_period).rsi()
        
        # 스토캐스틱
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.stoch_period,
            smooth_window=3
        )
        df['slowk'] = stoch.stoch()
        df['slowd'] = stoch.stoch_signal()
        
        # 밴드 폭
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_width_change'] = df['bb_width'].pct_change()
        
        # 캔들 색상
        df['is_bullish'] = df['close'] > df['open']
        
        return df
        
    def check_trend(self, df: pd.DataFrame) -> Dict[str, bool]:
        """추세를 판단합니다."""
        current_price = df['close'].iloc[-1]
        
        # 단기 추세 (20일선 vs 60일선)
        short_trend_up = df['ma_short'].iloc[-1] > df['ma_long'].iloc[-1]
        
        # 장기 추세 (현재가 vs 200일선)
        long_trend_up = current_price > df['ma_200'].iloc[-1]
        
        return {
            'short_trend_up': short_trend_up,
            'long_trend_up': long_trend_up
        }
        
    def check_band_conditions(self, df: pd.DataFrame) -> Dict[str, bool]:
        """볼린저 밴드 조건을 확인합니다."""
        current_price = df['close'].iloc[-1]
        
        # 밴드 터치
        upper_touch = current_price >= df['bb_upper'].iloc[-1]
        lower_touch = current_price <= df['bb_lower'].iloc[-1]
        
        # 밴드 스퀴즈
        squeeze = df['bb_width'].iloc[-1] < df['bb_width'].rolling(20).mean().iloc[-1] * 0.8
        
        # 밴드 확장
        expansion = df['bb_width_change'].iloc[-1] > 0.1
        
        return {
            'upper_touch': upper_touch,
            'lower_touch': lower_touch,
            'squeeze': squeeze,
            'expansion': expansion
        }
        
    def check_rsi_condition(self, df: pd.DataFrame) -> Dict[str, bool]:
        """RSI 조건을 확인합니다."""
        current_rsi = df['rsi'].iloc[-1]
        
        overbought = current_rsi >= 70
        oversold = current_rsi <= 30
        
        return {
            'overbought': overbought,
            'oversold': oversold
        }
        
    def check_stoch_condition(self, df: pd.DataFrame) -> Dict[str, bool]:
        """스토캐스틱 조건을 확인합니다."""
        current_k = df['slowk'].iloc[-1]
        
        overbought = current_k >= 80
        oversold = current_k <= 20
        
        return {
            'overbought': overbought,
            'oversold': oversold
        }
        
    def check_candle_change(self, df: pd.DataFrame) -> Dict[str, bool]:
        """캔들 색상 변화를 확인합니다."""
        current_candle = df['is_bullish'].iloc[-1]
        previous_candle = df['is_bullish'].iloc[-2]
        
        bear_to_bull = not previous_candle and current_candle
        bull_to_bear = previous_candle and not current_candle
        
        return {
            'bear_to_bull': bear_to_bull,
            'bull_to_bear': bull_to_bear
        }
        
    def find_abc_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """ABC 포인트를 찾습니다."""
        recent_df = df.tail(100)
        
        a_point = recent_df['high'].max()
        a_index = recent_df['high'].idxmax()
        
        after_a = recent_df.loc[a_index:]
        b_point = after_a['low'].min()
        b_index = after_a['low'].idxmin()
        
        after_b = recent_df.loc[b_index:]
        c_point = after_b['high'].max()
        c_index = after_b['high'].idxmax()
        
        after_c = recent_df.loc[c_index:]
        d_point = after_c['low'].min()
        
        return {
            'a': a_point,
            'b': b_point,
            'c': c_point,
            'd': d_point
        }
        
    def check_abc_pattern(self, df: pd.DataFrame, points: Dict[str, float]) -> bool:
        """ABC 패턴이 유효한지 확인합니다."""
        if points['c'] > points['a']:
            return False
            
        if points['d'] < points['b']:
            return False
            
        return True
        
    def find_trendlines(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """상승/하락 추세선을 찾고 신뢰도를 검증합니다."""
        recent_df = df.tail(self.trendline_lookback)
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        indices = np.arange(len(recent_df))
        
        # 상승 추세선 찾기 (저점 연결)
        support_points = []
        for i in range(2, len(lows)):
            if lows[i] > lows[i-1] and lows[i-1] > lows[i-2]:
                support_points.append((indices[i], lows[i]))
                
        # 하락 추세선 찾기 (고점 연결)
        resistance_points = []
        for i in range(2, len(highs)):
            if highs[i] < highs[i-1] and highs[i-1] < highs[i-2]:
                resistance_points.append((indices[i], highs[i]))
                
        # 추세선 신뢰도 계산
        support_line = self.calculate_trendline_with_confidence(support_points)
        resistance_line = self.calculate_trendline_with_confidence(resistance_points)
        
        return {
            'support': {
                'points': support_points,
                'line': support_line,
                'is_valid': len(support_points) >= self.min_touch_points and 
                           support_line['r_squared'] > 0.7 if support_line else False
            },
            'resistance': {
                'points': resistance_points,
                'line': resistance_line,
                'is_valid': len(resistance_points) >= self.min_touch_points and 
                           resistance_line['r_squared'] > 0.7 if resistance_line else False
            }
        }
        
    def calculate_trendline_with_confidence(self, points: List[Tuple[float, float]]) -> Dict:
        """추세선의 기울기, 절편, 신뢰도를 계산합니다."""
        if len(points) < 2:
            return None
            
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # R-squared 값 계산
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'std_err': std_err
        }
        
    def check_touch(self, df: pd.DataFrame, trendlines: Dict[str, Dict]) -> Dict[str, bool]:
        """현재 가격이 추세선에 터치했는지 확인합니다."""
        current_price = df['close'].iloc[-1]
        current_index = len(df) - 1
        
        touches = {
            'support': False,
            'resistance': False
        }
        
        # 지지선 터치 확인
        if trendlines['support']['is_valid']:
            slope, intercept = trendlines['support']['line']['slope'], trendlines['support']['line']['intercept']
            if slope and intercept:
                trendline_price = slope * current_index + intercept
                if abs(current_price - trendline_price) / trendline_price <= self.touch_threshold:
                    touches['support'] = True
                    
        # 저항선 터치 확인
        if trendlines['resistance']['is_valid']:
            slope, intercept = trendlines['resistance']['line']['slope'], trendlines['resistance']['line']['intercept']
            if slope and intercept:
                trendline_price = slope * current_index + intercept
                if abs(current_price - trendline_price) / trendline_price <= self.touch_threshold:
                    touches['resistance'] = True
                    
        return touches
        
    def check_band_touch(self, df: pd.DataFrame) -> Dict[str, bool]:
        """볼린저 밴드 터치를 확인합니다."""
        current_price = df['close'].iloc[-1]
        upper_band = df['bb_upper'].iloc[-1]
        lower_band = df['bb_lower'].iloc[-1]
        
        # 밴드 터치 판단 (0.2% 이내)
        upper_touch = abs(current_price - upper_band) / upper_band <= self.touch_threshold
        lower_touch = abs(current_price - lower_band) / lower_band <= self.touch_threshold
        
        # 밴드 돌파 확인 (0.2% 이상)
        upper_break = (current_price - upper_band) / upper_band > self.touch_threshold
        lower_break = (lower_band - current_price) / lower_band > self.touch_threshold
        
        return {
            'upper_touch': upper_touch,
            'lower_touch': lower_touch,
            'upper_break': upper_break,
            'lower_break': lower_break
        }
        
    def check_price_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        """가격 행동을 상세히 분석합니다."""
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        # 캔들 패턴 분석
        is_bullish = current_price > df['open'].iloc[-1]
        is_strong_bullish = current_price > df['high'].iloc[-2]
        is_strong_bearish = current_price < df['low'].iloc[-2]
        
        # 캔들 크기 분석
        candle_size = abs(current_price - df['open'].iloc[-1])
        avg_candle_size = abs(df['close'] - df['open']).rolling(20).mean().iloc[-1]
        is_large_candle = candle_size > avg_candle_size * 1.5
        
        # 거래량 분석
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        is_high_volume = current_volume > avg_volume * 1.5
        
        # 가격 모멘텀 분석
        price_change = (current_price - prev_price) / prev_price
        is_strong_move = abs(price_change) > self.touch_threshold * 2
        
        # RSI 모멘텀
        rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2]
        rsi_momentum = rsi - prev_rsi
        
        return {
            'is_bullish': is_bullish,
            'is_strong_bullish': is_strong_bullish,
            'is_strong_bearish': is_strong_bearish,
            'is_large_candle': is_large_candle,
            'is_high_volume': is_high_volume,
            'is_strong_move': is_strong_move,
            'rsi_momentum': rsi_momentum,
            'price_change': price_change
        }
        
    def check_signal_strength(self, df: pd.DataFrame, conditions: Dict) -> float:
        """신호의 강도를 계산합니다."""
        strength = 0.0
        
        # 추세선 신뢰도
        if conditions['trendline_touches']['support'] or conditions['trendline_touches']['resistance']:
            strength += 0.3
            
        # 볼린저 밴드 조건
        if conditions['band_touches']['upper_touch'] or conditions['band_touches']['lower_touch']:
            strength += 0.2
            
        # RSI 조건
        if conditions['rsi_condition']['oversold'] or conditions['rsi_condition']['overbought']:
            strength += 0.2
            
        # 가격 행동
        if conditions['price_action']['is_large_candle']:
            strength += 0.1
        if conditions['price_action']['is_high_volume']:
            strength += 0.1
        if conditions['price_action']['is_strong_move']:
            strength += 0.1
            
        return min(strength, 1.0)
        
    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """매매 신호를 생성합니다."""
        df = self.calculate_indicators(df)
        trend = self.check_trend(df)
        band_conditions = self.check_band_conditions(df)
        rsi_condition = self.check_rsi_condition(df)
        stoch_condition = self.check_stoch_condition(df)
        candle_change = self.check_candle_change(df)
        points = self.find_abc_points(df)
        
        # 추가된 분석
        trendlines = self.find_trendlines(df)
        trendline_touches = self.check_touch(df, trendlines)
        band_touches = self.check_band_touch(df)
        price_action = self.check_price_action(df)
        
        # 신호 강도 계산
        conditions = {
            'trendline_touches': trendline_touches,
            'band_touches': band_touches,
            'rsi_condition': rsi_condition,
            'price_action': price_action
        }
        signal_strength = self.check_signal_strength(df, conditions)
        
        signal = {
            'position': None,
            'reason': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'strength': signal_strength
        }
        
        current_price = df['close'].iloc[-1]
        
        # 상승추세 진입 조건
        if trend['long_trend_up']:
            # 저항선 터치 + RSI 과매도 + 강한 모멘텀
            if (trendline_touches['resistance'] and 
                rsi_condition['oversold'] and 
                price_action['is_bullish'] and
                price_action['rsi_momentum'] > 0 and
                price_action['is_high_volume']):
                signal['position'] = 'long'
                signal['reason'] = "상승추세 + 저항선 터치 + RSI 과매도 + 강한 모멘텀"
                
            # 볼린저 밴드 하단 터치 + 강한 반등 + 거래량 증가
            elif (band_touches['lower_touch'] and 
                  price_action['is_strong_bullish'] and
                  price_action['is_high_volume']):
                signal['position'] = 'long'
                signal['reason'] = "상승추세 + 볼린저 밴드 하단 터치 + 강한 반등 + 거래량 증가"
                
        # 하락추세 진입 조건
        elif not trend['long_trend_up']:
            # 지지선 터치 + RSI 과매수 + 강한 하락 모멘텀
            if (trendline_touches['support'] and 
                rsi_condition['overbought'] and 
                price_action['is_strong_bearish'] and
                price_action['rsi_momentum'] < 0 and
                price_action['is_high_volume']):
                signal['position'] = 'short'
                signal['reason'] = "하락추세 + 지지선 터치 + RSI 과매수 + 강한 하락 모멘텀"
                
            # 볼린저 밴드 상단 터치 + 강한 하락 + 거래량 증가
            elif (band_touches['upper_touch'] and 
                  price_action['is_strong_bearish'] and
                  price_action['is_high_volume']):
                signal['position'] = 'short'
                signal['reason'] = "하락추세 + 볼린저 밴드 상단 터치 + 강한 하락 + 거래량 증가"
                
        if signal['position'] is not None:
            signal['entry_price'] = current_price
            if signal['position'] == 'long':
                signal['stop_loss'] = current_price * (1 - self.stop_loss_ratio)
                signal['take_profit'] = current_price * (1 + self.take_profit_ratio)
            else:
                signal['stop_loss'] = current_price * (1 + self.stop_loss_ratio)
                signal['take_profit'] = current_price * (1 - self.take_profit_ratio)
                
        return signal 