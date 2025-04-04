"""
패턴 인식 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.visualization.advanced_charts import AdvancedCharts

class PatternRecognizer:
    """패턴 인식 클래스"""
    
    def __init__(self, db_manager):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.charts = AdvancedCharts()
        self.scaler = StandardScaler()
        self.patterns = {}
        
    def identify_candlestick_patterns(self,
                                    data: pd.DataFrame,
                                    window: int = 5) -> List[Dict]:
        """
        캔들스틱 패턴 인식
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            window (int): 패턴 윈도우 크기
            
        Returns:
            List[Dict]: 인식된 패턴 목록
        """
        try:
            patterns = []
            
            for i in range(window, len(data)):
                window_data = data.iloc[i-window:i+1]
                
                # 상승 추세 확인
                is_uptrend = all(window_data['close'] > window_data['open'])
                
                # 하락 추세 확인
                is_downtrend = all(window_data['close'] < window_data['open'])
                
                # 도지 패턴 확인
                is_doji = abs(window_data['close'].iloc[-1] - window_data['open'].iloc[-1]) <= \
                         (window_data['high'].iloc[-1] - window_data['low'].iloc[-1]) * 0.1
                
                # 해머 패턴 확인
                is_hammer = (window_data['high'].iloc[-1] - window_data['low'].iloc[-1]) > \
                           (window_data['high'].iloc[-1] - window_data['low'].iloc[-1]).mean() * 1.5 and \
                           abs(window_data['close'].iloc[-1] - window_data['low'].iloc[-1]) > \
                           abs(window_data['high'].iloc[-1] - window_data['close'].iloc[-1]) * 2
                
                # 샛별 패턴 확인
                is_shooting_star = (window_data['high'].iloc[-1] - window_data['low'].iloc[-1]) > \
                                  (window_data['high'].iloc[-1] - window_data['low'].iloc[-1]).mean() * 1.5 and \
                                  abs(window_data['high'].iloc[-1] - window_data['close'].iloc[-1]) > \
                                  abs(window_data['close'].iloc[-1] - window_data['low'].iloc[-1]) * 2
                
                # 패턴 저장
                if is_uptrend:
                    patterns.append({
                        'start_time': window_data.index[0],
                        'end_time': window_data.index[-1],
                        'name': '상승 추세',
                        'type': 'trend',
                        'strength': self._calculate_pattern_strength(window_data)
                    })
                    
                if is_downtrend:
                    patterns.append({
                        'start_time': window_data.index[0],
                        'end_time': window_data.index[-1],
                        'name': '하락 추세',
                        'type': 'trend',
                        'strength': self._calculate_pattern_strength(window_data)
                    })
                    
                if is_doji:
                    patterns.append({
                        'start_time': window_data.index[-1],
                        'end_time': window_data.index[-1],
                        'name': '도지',
                        'type': 'reversal',
                        'strength': self._calculate_pattern_strength(window_data)
                    })
                    
                if is_hammer:
                    patterns.append({
                        'start_time': window_data.index[-1],
                        'end_time': window_data.index[-1],
                        'name': '해머',
                        'type': 'reversal',
                        'strength': self._calculate_pattern_strength(window_data)
                    })
                    
                if is_shooting_star:
                    patterns.append({
                        'start_time': window_data.index[-1],
                        'end_time': window_data.index[-1],
                        'name': '샛별',
                        'type': 'reversal',
                        'strength': self._calculate_pattern_strength(window_data)
                    })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"캔들스틱 패턴 인식 실패: {str(e)}")
            return []
            
    def identify_price_patterns(self,
                              data: pd.DataFrame,
                              window: int = 20) -> List[Dict]:
        """
        가격 패턴 인식
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            window (int): 패턴 윈도우 크기
            
        Returns:
            List[Dict]: 인식된 패턴 목록
        """
        try:
            patterns = []
            
            for i in range(window, len(data)):
                window_data = data.iloc[i-window:i+1]
                
                # 이중 바닥 패턴 확인
                is_double_bottom = self._check_double_bottom(window_data)
                
                # 이중 천장 패턴 확인
                is_double_top = self._check_double_top(window_data)
                
                # 헤드 앤 숄더 패턴 확인
                is_head_and_shoulders = self._check_head_and_shoulders(window_data)
                
                # 역 헤드 앤 숄더 패턴 확인
                is_inverse_head_and_shoulders = self._check_inverse_head_and_shoulders(window_data)
                
                # 패턴 저장
                if is_double_bottom:
                    patterns.append({
                        'start_time': window_data.index[0],
                        'end_time': window_data.index[-1],
                        'name': '이중 바닥',
                        'type': 'reversal',
                        'strength': self._calculate_pattern_strength(window_data)
                    })
                    
                if is_double_top:
                    patterns.append({
                        'start_time': window_data.index[0],
                        'end_time': window_data.index[-1],
                        'name': '이중 천장',
                        'type': 'reversal',
                        'strength': self._calculate_pattern_strength(window_data)
                    })
                    
                if is_head_and_shoulders:
                    patterns.append({
                        'start_time': window_data.index[0],
                        'end_time': window_data.index[-1],
                        'name': '헤드 앤 숄더',
                        'type': 'reversal',
                        'strength': self._calculate_pattern_strength(window_data)
                    })
                    
                if is_inverse_head_and_shoulders:
                    patterns.append({
                        'start_time': window_data.index[0],
                        'end_time': window_data.index[-1],
                        'name': '역 헤드 앤 숄더',
                        'type': 'reversal',
                        'strength': self._calculate_pattern_strength(window_data)
                    })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"가격 패턴 인식 실패: {str(e)}")
            return []
            
    def _check_double_bottom(self, data: pd.DataFrame) -> bool:
        """
        이중 바닥 패턴 확인
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            bool: 이중 바닥 패턴 여부
        """
        try:
            # 첫 번째 바닥
            first_bottom = data['low'].iloc[:len(data)//2].min()
            first_bottom_idx = data['low'].iloc[:len(data)//2].idxmin()
            
            # 두 번째 바닥
            second_bottom = data['low'].iloc[len(data)//2:].min()
            second_bottom_idx = data['low'].iloc[len(data)//2:].idxmin()
            
            # 바닥 간 차이 확인
            bottom_diff = abs(first_bottom - second_bottom) / first_bottom
            
            # 목선 확인
            neckline = max(data['high'].iloc[first_bottom_idx:second_bottom_idx])
            
            return bottom_diff < 0.02 and second_bottom_idx > first_bottom_idx
            
        except Exception as e:
            self.logger.error(f"이중 바닥 패턴 확인 실패: {str(e)}")
            return False
            
    def _check_double_top(self, data: pd.DataFrame) -> bool:
        """
        이중 천장 패턴 확인
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            bool: 이중 천장 패턴 여부
        """
        try:
            # 첫 번째 천장
            first_top = data['high'].iloc[:len(data)//2].max()
            first_top_idx = data['high'].iloc[:len(data)//2].idxmax()
            
            # 두 번째 천장
            second_top = data['high'].iloc[len(data)//2:].max()
            second_top_idx = data['high'].iloc[len(data)//2:].idxmax()
            
            # 천장 간 차이 확인
            top_diff = abs(first_top - second_top) / first_top
            
            # 목선 확인
            neckline = min(data['low'].iloc[first_top_idx:second_top_idx])
            
            return top_diff < 0.02 and second_top_idx > first_top_idx
            
        except Exception as e:
            self.logger.error(f"이중 천장 패턴 확인 실패: {str(e)}")
            return False
            
    def _check_head_and_shoulders(self, data: pd.DataFrame) -> bool:
        """
        헤드 앤 숄더 패턴 확인
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            bool: 헤드 앤 숄더 패턴 여부
        """
        try:
            # 왼쪽 어깨
            left_shoulder = data['high'].iloc[:len(data)//3].max()
            left_shoulder_idx = data['high'].iloc[:len(data)//3].idxmax()
            
            # 머리
            head = data['high'].iloc[len(data)//3:2*len(data)//3].max()
            head_idx = data['high'].iloc[len(data)//3:2*len(data)//3].idxmax()
            
            # 오른쪽 어깨
            right_shoulder = data['high'].iloc[2*len(data)//3:].max()
            right_shoulder_idx = data['high'].iloc[2*len(data)//3:].idxmax()
            
            # 목선 확인
            neckline = min(data['low'].iloc[left_shoulder_idx:right_shoulder_idx])
            
            return (head > left_shoulder and head > right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.02)
            
        except Exception as e:
            self.logger.error(f"헤드 앤 숄더 패턴 확인 실패: {str(e)}")
            return False
            
    def _check_inverse_head_and_shoulders(self, data: pd.DataFrame) -> bool:
        """
        역 헤드 앤 숄더 패턴 확인
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            bool: 역 헤드 앤 숄더 패턴 여부
        """
        try:
            # 왼쪽 어깨
            left_shoulder = data['low'].iloc[:len(data)//3].min()
            left_shoulder_idx = data['low'].iloc[:len(data)//3].idxmin()
            
            # 머리
            head = data['low'].iloc[len(data)//3:2*len(data)//3].min()
            head_idx = data['low'].iloc[len(data)//3:2*len(data)//3].idxmin()
            
            # 오른쪽 어깨
            right_shoulder = data['low'].iloc[2*len(data)//3:].min()
            right_shoulder_idx = data['low'].iloc[2*len(data)//3:].idxmin()
            
            # 목선 확인
            neckline = max(data['high'].iloc[left_shoulder_idx:right_shoulder_idx])
            
            return (head < left_shoulder and head < right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.02)
            
        except Exception as e:
            self.logger.error(f"역 헤드 앤 숄더 패턴 확인 실패: {str(e)}")
            return False
            
    def _calculate_pattern_strength(self, data: pd.DataFrame) -> float:
        """
        패턴 강도 계산
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            float: 패턴 강도
        """
        try:
            # 가격 변동성
            volatility = data['close'].std() / data['close'].mean()
            
            # 거래량 확인
            volume_ratio = data['volume'].iloc[-1] / data['volume'].mean()
            
            # 패턴 강도 계산
            strength = (1 - volatility) * volume_ratio
            
            return np.clip(strength, 0, 1)
            
        except Exception as e:
            self.logger.error(f"패턴 강도 계산 실패: {str(e)}")
            return 0.0
            
    def analyze_pattern_performance(self,
                                  patterns: List[Dict],
                                  trades: List[Dict]) -> Dict:
        """
        패턴 성과 분석
        
        Args:
            patterns (List[Dict]): 인식된 패턴
            trades (List[Dict]): 거래 기록
            
        Returns:
            Dict: 패턴별 성과 분석 결과
        """
        try:
            pattern_performance = {}
            
            for pattern in patterns:
                # 패턴 기간 내 거래 필터링
                pattern_trades = [
                    t for t in trades
                    if pattern['start_time'] <= t['entry_time'] <= pattern['end_time']
                ]
                
                if pattern_trades:
                    # 패턴별 성과 계산
                    total_trades = len(pattern_trades)
                    winning_trades = len([t for t in pattern_trades if t['pnl'] > 0])
                    total_pnl = sum(t['pnl'] for t in pattern_trades)
                    avg_pnl = total_pnl / total_trades
                    win_rate = winning_trades / total_trades
                    
                    pattern_performance[pattern['name']] = {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'total_pnl': total_pnl,
                        'avg_pnl': avg_pnl,
                        'win_rate': win_rate,
                        'pattern_strength': pattern['strength']
                    }
            
            return pattern_performance
            
        except Exception as e:
            self.logger.error(f"패턴 성과 분석 실패: {str(e)}")
            return {}
            
    def generate_pattern_report(self,
                              start_time: datetime,
                              end_time: datetime) -> Dict:
        """
        패턴 분석 리포트 생성
        
        Args:
            start_time (datetime): 시작 시간
            end_time (datetime): 종료 시간
            
        Returns:
            Dict: 패턴 분석 리포트
        """
        try:
            # 시장 데이터 조회
            market_data = self.db_manager.get_market_data(
                start_time=start_time,
                end_time=end_time
            )
            
            # 거래 기록 조회
            trades = self.db_manager.get_trades(
                start_time=start_time,
                end_time=end_time
            )
            
            # 패턴 인식
            candlestick_patterns = self.identify_candlestick_patterns(market_data)
            price_patterns = self.identify_price_patterns(market_data)
            
            # 패턴 성과 분석
            candlestick_performance = self.analyze_pattern_performance(candlestick_patterns, trades)
            price_performance = self.analyze_pattern_performance(price_patterns, trades)
            
            # 패턴 차트 생성
            pattern_chart = self.charts.create_pattern_chart(market_data, candlestick_patterns + price_patterns)
            
            return {
                'candlestick_patterns': candlestick_patterns,
                'price_patterns': price_patterns,
                'candlestick_performance': candlestick_performance,
                'price_performance': price_performance,
                'pattern_chart': pattern_chart
            }
            
        except Exception as e:
            self.logger.error(f"패턴 분석 리포트 생성 실패: {str(e)}")
            return None 