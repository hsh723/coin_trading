import pandas as pd
from typing import Dict, Any, List, Optional
from ..utils.logger import setup_logger
import logging
import numpy as np
from ta.trend import SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from ..utils.database import DatabaseManager
import ta

class TechnicalAnalyzer:
    """기술적 분석기 클래스"""
    
    def __init__(self, db: DatabaseManager):
        """기술적 분석기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.db = db
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        try:
            # 이동평균선
            df['ma20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['ma50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ma200'] = ta.trend.sma_indicator(df['close'], window=200)
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            # 볼린저 밴드
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # ADX
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            
            # ATR
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 실패: {str(e)}")
            return df
            
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """변동성 계산"""
        try:
            return df['atr'].iloc[-1] / df['close'].iloc[-1]
        except Exception as e:
            self.logger.error(f"변동성 계산 실패: {str(e)}")
            return 0.0
            
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """추세 강도 계산"""
        try:
            return df['adx'].iloc[-1] / 100.0
        except Exception as e:
            self.logger.error(f"추세 강도 계산 실패: {str(e)}")
            return 0.0
            
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """거래 신호 생성"""
        try:
            current_price = df['close'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            adx = df['adx'].iloc[-1]
            
            # 신호 강도 계산
            signal_strength = 0.0
            signal = 'neutral'
            
            # RSI 기반 신호
            if rsi < 30:  # 과매도
                signal_strength += 0.3
                signal = 'buy'
            elif rsi > 70:  # 과매수
                signal_strength -= 0.3
                signal = 'sell'
                
            # MACD 기반 신호
            if macd > macd_signal:  # 골든크로스
                signal_strength += 0.3
                signal = 'buy'
            elif macd < macd_signal:  # 데드크로스
                signal_strength -= 0.3
                signal = 'sell'
                
            # ADX 기반 신호 강화
            if adx > 25:  # 강한 추세
                signal_strength *= 1.2
                
            return {
                'signal': signal,
                'strength': abs(signal_strength)
            }
            
        except Exception as e:
            self.logger.error(f"거래 신호 생성 실패: {str(e)}")
            return {'signal': 'neutral', 'strength': 0.0}
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """모든 기술적 지표 계산"""
        indicators = {
            'trend': self._calculate_trend_indicators(data),
            'momentum': self._calculate_momentum_indicators(data),
            'volatility': self._calculate_volatility_indicators(data),
            'volume': self._calculate_volume_indicators(data),
            'oscillators': self._calculate_oscillators(data),
            'ichimoku': self._calculate_ichimoku(data),
            'fibonacci': self._calculate_fibonacci(data),
            'pivot_points': self._calculate_pivot_points(data),
            'market_profile': self._calculate_market_profile(data)
        }
        return indicators
        
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict:
        """추세 지표 계산"""
        # 이동 평균선
        ma20 = SMAIndicator(close=data['close'], window=20).sma_indicator()
        ma50 = SMAIndicator(close=data['close'], window=50).sma_indicator()
        ma200 = SMAIndicator(close=data['close'], window=200).sma_indicator()
        
        # MACD
        macd_ind = MACD(close=data['close'])
        macd = macd_ind.macd()
        macd_signal = macd_ind.macd_signal()
        macd_hist = macd_ind.macd_diff()
        
        # ADX
        adx_ind = ADXIndicator(high=data['high'], low=data['low'], close=data['close'])
        adx = adx_ind.adx()
        
        return {
            'ma20': ma20,
            'ma50': ma50,
            'ma200': ma200,
            'macd': {
                'macd': macd,
                'signal': macd_signal,
                'histogram': macd_hist
            },
            'adx': adx
        }
        
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict:
        """모멘텀 지표 계산"""
        # RSI
        rsi = RSIIndicator(close=data['close']).rsi()
        
        # 스토캐스틱
        stoch = StochasticOscillator(
            high=data['high'],
            low=data['low'],
            close=data['close']
        )
        stoch_k = stoch.stoch()
        stoch_d = stoch.stoch_signal()
        
        return {
            'rsi': rsi,
            'stochastic': {
                'k': stoch_k,
                'd': stoch_d
            }
        }
        
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict:
        """변동성 지표 계산"""
        # 볼린저 밴드
        bb = BollingerBands(close=data['close'])
        
        # ATR
        atr = AverageTrueRange(
            high=data['high'],
            low=data['low'],
            close=data['close']
        ).average_true_range()
        
        return {
            'bollinger': {
                'upper': bb.bollinger_hband(),
                'middle': bb.bollinger_mavg(),
                'lower': bb.bollinger_lband()
            },
            'atr': atr
        }
        
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict:
        """거래량 지표 계산"""
        # VWAP
        vwap = VolumeWeightedAveragePrice(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume']
        ).volume_weighted_average_price()
        
        # 거래량 이동평균
        volume_ma = SMAIndicator(close=data['volume'], window=20).sma_indicator()
        
        return {
            'vwap': vwap,
            'volume_ma': volume_ma
        }
        
    def _calculate_oscillators(self, data: pd.DataFrame) -> Dict:
        """오실레이터 계산"""
        # 윌리엄스 %R
        willr = abstract.WILLR(data)
        
        # ROC
        roc = abstract.ROC(data)
        
        # MOM
        mom = abstract.MOM(data)
        
        return {
            'willr': willr,
            'roc': roc,
            'mom': mom
        }
        
    def _calculate_ichimoku(self, data: pd.DataFrame) -> Dict:
        """일목균형표 계산"""
        # 전환선
        conversion_line = (data['high'].rolling(window=9).max() + 
                         data['low'].rolling(window=9).min()) / 2
        
        # 기준선
        base_line = (data['high'].rolling(window=26).max() + 
                    data['low'].rolling(window=26).min()) / 2
        
        # 선행스팬1
        leading_span1 = (conversion_line + base_line) / 2
        
        # 선행스팬2
        leading_span2 = (data['high'].rolling(window=52).max() + 
                        data['low'].rolling(window=52).min()) / 2
        
        # 후행스팬
        lagging_span = data['close'].shift(-26)
        
        return {
            'conversion_line': conversion_line,
            'base_line': base_line,
            'leading_span1': leading_span1,
            'leading_span2': leading_span2,
            'lagging_span': lagging_span
        }
        
    def _calculate_fibonacci(self, data: pd.DataFrame) -> Dict:
        """피보나치 레벨 계산"""
        high = data['high'].max()
        low = data['low'].min()
        diff = high - low
        
        levels = {
            '0': low,
            '0.236': low + diff * 0.236,
            '0.382': low + diff * 0.382,
            '0.5': low + diff * 0.5,
            '0.618': low + diff * 0.618,
            '0.786': low + diff * 0.786,
            '1': high
        }
        
        return levels
        
    def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict:
        """피봇 포인트 계산"""
        high = data['high'].iloc[-1]
        low = data['low'].iloc[-1]
        close = data['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'resistance': {
                'r1': r1,
                'r2': r2,
                'r3': r3
            },
            'support': {
                's1': s1,
                's2': s2,
                's3': s3
            }
        }
        
    def _calculate_market_profile(self, data: pd.DataFrame) -> Dict:
        """마켓 프로파일 계산"""
        # 가격 구간 설정
        price_range = np.linspace(data['low'].min(), data['high'].max(), 100)
        
        # 각 가격 구간별 거래량 계산
        profile = {}
        for i in range(len(price_range) - 1):
            mask = (data['close'] >= price_range[i]) & (data['close'] < price_range[i+1])
            profile[f"{price_range[i]:.2f}-{price_range[i+1]:.2f}"] = {
                'volume': data[mask]['volume'].sum(),
                'trades': len(data[mask]),
                'time': data[mask].index
            }
            
        # 가치 영역 계산
        total_volume = sum(v['volume'] for v in profile.values())
        target_volume = total_volume * 0.7
        
        sorted_profile = sorted(profile.items(), key=lambda x: x[1]['volume'], reverse=True)
        value_area_volume = 0
        value_area = []
        
        for level, data in sorted_profile:
            if value_area_volume < target_volume:
                value_area_volume += data['volume']
                value_area.append(level)
            else:
                break
                
        return {
            'profile': profile,
            'value_area': {
                'levels': value_area,
                'volume': value_area_volume,
                'percentage': value_area_volume / total_volume * 100
            }
        }
        
    def analyze_signals(self, data: pd.DataFrame) -> Dict:
        """매매 신호 분석"""
        indicators = self.calculate_indicators(data)
        
        signals = {
            'trend_signals': self._analyze_trend_signals(indicators['trend']),
            'momentum_signals': self._analyze_momentum_signals(indicators['momentum']),
            'pattern_signals': self._analyze_pattern_signals(data),
            'volume_signals': self._analyze_volume_signals(indicators['volume'])
        }
        
        return signals
        
    def _analyze_trend_signals(self, trend_indicators: Dict) -> Dict:
        """추세 신호 분석"""
        signals = {
            'ma_cross': None,
            'macd_cross': None,
            'adx_strength': None
        }
        
        # 이동평균선 크로스
        if trend_indicators['ma20'].iloc[-1] > trend_indicators['ma50'].iloc[-1]:
            signals['ma_cross'] = 'bullish'
        else:
            signals['ma_cross'] = 'bearish'
            
        # MACD 크로스
        if trend_indicators['macd']['macd'].iloc[-1] > trend_indicators['macd']['signal'].iloc[-1]:
            signals['macd_cross'] = 'bullish'
        else:
            signals['macd_cross'] = 'bearish'
            
        # ADX 강도
        if trend_indicators['adx'].iloc[-1] > 25:
            signals['adx_strength'] = 'strong'
        else:
            signals['adx_strength'] = 'weak'
            
        return signals
        
    def _analyze_momentum_signals(self, momentum_indicators: Dict) -> Dict:
        """모멘텀 신호 분석"""
        signals = {
            'rsi_signal': None,
            'stochastic_signal': None,
            'cci_signal': None
        }
        
        # RSI 신호
        if momentum_indicators['rsi'].iloc[-1] < 30:
            signals['rsi_signal'] = 'oversold'
        elif momentum_indicators['rsi'].iloc[-1] > 70:
            signals['rsi_signal'] = 'overbought'
        else:
            signals['rsi_signal'] = 'neutral'
            
        # 스토캐스틱 신호
        if (momentum_indicators['stochastic']['k'].iloc[-1] < 20 and 
            momentum_indicators['stochastic']['d'].iloc[-1] < 20):
            signals['stochastic_signal'] = 'oversold'
        elif (momentum_indicators['stochastic']['k'].iloc[-1] > 80 and 
              momentum_indicators['stochastic']['d'].iloc[-1] > 80):
            signals['stochastic_signal'] = 'overbought'
        else:
            signals['stochastic_signal'] = 'neutral'
            
        # CCI 신호
        if momentum_indicators['cci'].iloc[-1] < -100:
            signals['cci_signal'] = 'oversold'
        elif momentum_indicators['cci'].iloc[-1] > 100:
            signals['cci_signal'] = 'overbought'
        else:
            signals['cci_signal'] = 'neutral'
            
        return signals
        
    def _analyze_pattern_signals(self, data: pd.DataFrame) -> Dict:
        """패턴 신호 분석"""
        # 차트 패턴 분석
        chart_analyzer = ChartAnalyzer(self.db)
        patterns = chart_analyzer.detect_patterns(data)
        
        signals = {
            'pattern_type': None,
            'pattern_direction': None,
            'pattern_strength': None
        }
        
        # 가장 최근 패턴 찾기
        latest_pattern = None
        latest_end = None
        
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if latest_end is None or pattern['end'] > latest_end:
                    latest_end = pattern['end']
                    latest_pattern = pattern
                    
        if latest_pattern:
            signals['pattern_type'] = latest_pattern['type']
            signals['pattern_direction'] = latest_pattern.get('direction', 'neutral')
            signals['pattern_strength'] = 'strong' if latest_pattern.get('strength', 0) > 0.7 else 'weak'
            
        return signals
        
    def _analyze_volume_signals(self, volume_indicators: Dict) -> Dict:
        """거래량 신호 분석"""
        signals = {
            'obv_trend': None,
            'mfi_signal': None
        }
        
        # OBV 트렌드
        if volume_indicators['obv'].iloc[-1] > volume_indicators['obv'].iloc[-5]:
            signals['obv_trend'] = 'bullish'
        else:
            signals['obv_trend'] = 'bearish'
            
        # MFI 신호
        if volume_indicators['mfi'].iloc[-1] < 20:
            signals['mfi_signal'] = 'oversold'
        elif volume_indicators['mfi'].iloc[-1] > 80:
            signals['mfi_signal'] = 'overbought'
        else:
            signals['mfi_signal'] = 'neutral'
            
        return signals

    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """
        지지선과 저항선 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            window (int): 윈도우 크기
            
        Returns:
            Dict[str, List[float]]: 지지선과 저항선
        """
        try:
            # 피봇 포인트 계산
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['r1'] = 2 * df['pivot'] - df['low']
            df['s1'] = 2 * df['pivot'] - df['high']
            
            # 최근 지지선과 저항선
            recent_support = df['s1'].tail(window).tolist()
            recent_resistance = df['r1'].tail(window).tolist()
            
            return {
                'support': recent_support,
                'resistance': recent_resistance
            }
            
        except Exception as e:
            self.logger.error(f"지지선/저항선 계산 실패: {str(e)}")
            return {'support': [], 'resistance': []} 