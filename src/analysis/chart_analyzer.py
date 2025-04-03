import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from database import Database

class ChartAnalyzer:
    def __init__(self, db: Database):
        self.logger = logging.getLogger(__name__)
        self.db = db
        
    def detect_patterns(self, data: pd.DataFrame) -> Dict:
        """차트 패턴 감지"""
        patterns = {
            'head_and_shoulders': self._detect_head_and_shoulders(data),
            'double_top': self._detect_double_top(data),
            'double_bottom': self._detect_double_bottom(data),
            'triangles': self._detect_triangles(data),
            'flags': self._detect_flags(data),
            'wedges': self._detect_wedges(data)
        }
        return patterns
        
    def _detect_head_and_shoulders(self, data: pd.DataFrame) -> List[Dict]:
        """머리와 어깨 패턴 감지"""
        patterns = []
        window = 20
        
        for i in range(window, len(data) - window):
            left_shoulder = data['high'].iloc[i-window:i].max()
            head = data['high'].iloc[i:i+window].max()
            right_shoulder = data['high'].iloc[i+window:i+2*window].max()
            
            neckline = (data['low'].iloc[i-window:i].min() + 
                       data['low'].iloc[i+window:i+2*window].min()) / 2
            
            if (left_shoulder < head and right_shoulder < head and
                abs(left_shoulder - right_shoulder) < 0.02 * head):
                patterns.append({
                    'type': 'head_and_shoulders',
                    'start': data.index[i-window],
                    'end': data.index[i+2*window],
                    'neckline': neckline,
                    'target': neckline - (head - neckline)
                })
                
        return patterns
        
    def _detect_double_top(self, data: pd.DataFrame) -> List[Dict]:
        """이중 천정 패턴 감지"""
        patterns = []
        window = 10
        
        for i in range(window, len(data) - window):
            first_top = data['high'].iloc[i-window:i].max()
            second_top = data['high'].iloc[i:i+window].max()
            valley = data['low'].iloc[i-window:i+window].min()
            
            if (abs(first_top - second_top) < 0.02 * first_top and
                valley < first_top * 0.95):
                patterns.append({
                    'type': 'double_top',
                    'start': data.index[i-window],
                    'end': data.index[i+window],
                    'neckline': valley,
                    'target': valley - (first_top - valley)
                })
                
        return patterns
        
    def _detect_double_bottom(self, data: pd.DataFrame) -> List[Dict]:
        """이중 바닥 패턴 감지"""
        patterns = []
        window = 10
        
        for i in range(window, len(data) - window):
            first_bottom = data['low'].iloc[i-window:i].min()
            second_bottom = data['low'].iloc[i:i+window].min()
            peak = data['high'].iloc[i-window:i+window].max()
            
            if (abs(first_bottom - second_bottom) < 0.02 * first_bottom and
                peak > first_bottom * 1.05):
                patterns.append({
                    'type': 'double_bottom',
                    'start': data.index[i-window],
                    'end': data.index[i+window],
                    'neckline': peak,
                    'target': peak + (peak - first_bottom)
                })
                
        return patterns
        
    def _detect_triangles(self, data: pd.DataFrame) -> List[Dict]:
        """삼각형 패턴 감지"""
        patterns = []
        window = 20
        
        for i in range(window, len(data) - window):
            highs = data['high'].iloc[i-window:i+window]
            lows = data['low'].iloc[i-window:i+window]
            
            # 상승 추세선
            upper_trend = np.polyfit(range(len(highs)), highs, 1)
            # 하락 추세선
            lower_trend = np.polyfit(range(len(lows)), lows, 1)
            
            # 삼각형 각도 계산
            angle = np.arctan(abs(upper_trend[0] - lower_trend[0])) * 180 / np.pi
            
            if angle < 30:  # 삼각형 패턴으로 간주
                patterns.append({
                    'type': 'triangle',
                    'start': data.index[i-window],
                    'end': data.index[i+window],
                    'angle': angle,
                    'direction': 'ascending' if upper_trend[0] > lower_trend[0] else 'descending'
                })
                
        return patterns
        
    def _detect_flags(self, data: pd.DataFrame) -> List[Dict]:
        """깃발 패턴 감지"""
        patterns = []
        window = 10
        
        for i in range(window, len(data) - window):
            # 깃대 부분
            pole_high = data['high'].iloc[i-2*window:i-window].max()
            pole_low = data['low'].iloc[i-2*window:i-window].min()
            
            # 깃발 부분
            flag_high = data['high'].iloc[i-window:i].max()
            flag_low = data['low'].iloc[i-window:i].min()
            
            if (pole_high - pole_low > 0.05 * pole_high and  # 깃대가 충분히 크고
                flag_high - flag_low < 0.02 * flag_high):   # 깃발이 충분히 작으면
                patterns.append({
                    'type': 'flag',
                    'start': data.index[i-2*window],
                    'end': data.index[i],
                    'pole_height': pole_high - pole_low,
                    'direction': 'bullish' if data['close'].iloc[i] > data['close'].iloc[i-window] else 'bearish'
                })
                
        return patterns
        
    def _detect_wedges(self, data: pd.DataFrame) -> List[Dict]:
        """쐐기 패턴 감지"""
        patterns = []
        window = 20
        
        for i in range(window, len(data) - window):
            highs = data['high'].iloc[i-window:i+window]
            lows = data['low'].iloc[i-window:i+window]
            
            # 상승 추세선
            upper_trend = np.polyfit(range(len(highs)), highs, 1)
            # 하락 추세선
            lower_trend = np.polyfit(range(len(lows)), lows, 1)
            
            # 쐐기 각도 계산
            angle = np.arctan(abs(upper_trend[0] - lower_trend[0])) * 180 / np.pi
            
            if 30 < angle < 60:  # 쐐기 패턴으로 간주
                patterns.append({
                    'type': 'wedge',
                    'start': data.index[i-window],
                    'end': data.index[i+window],
                    'angle': angle,
                    'direction': 'rising' if upper_trend[0] > 0 else 'falling'
                })
                
        return patterns
        
    def analyze_volume_profile(self, data: pd.DataFrame) -> Dict:
        """거래량 프로파일 분석"""
        # 가격 구간 설정
        price_range = np.linspace(data['low'].min(), data['high'].max(), 100)
        
        # 각 가격 구간별 거래량 계산
        volume_profile = {}
        for i in range(len(price_range) - 1):
            mask = (data['close'] >= price_range[i]) & (data['close'] < price_range[i+1])
            volume_profile[f"{price_range[i]:.2f}-{price_range[i+1]:.2f}"] = data[mask]['volume'].sum()
            
        # 주요 가격 레벨 찾기
        sorted_profile = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        major_levels = sorted_profile[:5]
        
        return {
            'volume_profile': volume_profile,
            'major_levels': major_levels,
            'value_area': self._calculate_value_area(volume_profile)
        }
        
    def _calculate_value_area(self, volume_profile: Dict) -> Dict:
        """가치 영역 계산"""
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * 0.7  # 70% 거래량
        
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        value_area_volume = 0
        value_area_levels = []
        
        for level, volume in sorted_levels:
            if value_area_volume < target_volume:
                value_area_volume += volume
                value_area_levels.append(level)
            else:
                break
                
        return {
            'levels': value_area_levels,
            'volume': value_area_volume,
            'percentage': value_area_volume / total_volume * 100
        }
        
    def analyze_support_resistance(self, data: pd.DataFrame) -> Dict:
        """지지/저항 레벨 분석"""
        window = 20
        levels = {
            'support': [],
            'resistance': []
        }
        
        # 이동 평균선 기반 지지/저항
        ma20 = data['close'].rolling(window=window).mean()
        ma50 = data['close'].rolling(window=50).mean()
        ma200 = data['close'].rolling(window=200).mean()
        
        levels['support'].extend([
            {'level': ma20.iloc[-1], 'strength': 1, 'type': 'ma20'},
            {'level': ma50.iloc[-1], 'strength': 2, 'type': 'ma50'},
            {'level': ma200.iloc[-1], 'strength': 3, 'type': 'ma200'}
        ])
        
        # 피봇 포인트 기반 지지/저항
        pivot = (data['high'].iloc[-1] + data['low'].iloc[-1] + data['close'].iloc[-1]) / 3
        r1 = 2 * pivot - data['low'].iloc[-1]
        s1 = 2 * pivot - data['high'].iloc[-1]
        
        levels['resistance'].append({'level': r1, 'strength': 2, 'type': 'pivot'})
        levels['support'].append({'level': s1, 'strength': 2, 'type': 'pivot'})
        
        return levels
        
    def analyze_trend(self, data: pd.DataFrame) -> Dict:
        """추세 분석"""
        # 이동 평균선
        ma20 = data['close'].rolling(window=20).mean()
        ma50 = data['close'].rolling(window=50).mean()
        ma200 = data['close'].rolling(window=200).mean()
        
        # 추세 강도
        trend_strength = abs(data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
        
        # 추세 방향
        if ma20.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1]:
            direction = 'uptrend'
        elif ma20.iloc[-1] < ma50.iloc[-1] < ma200.iloc[-1]:
            direction = 'downtrend'
        else:
            direction = 'sideways'
            
        return {
            'direction': direction,
            'strength': trend_strength,
            'ma20': ma20.iloc[-1],
            'ma50': ma50.iloc[-1],
            'ma200': ma200.iloc[-1]
        } 