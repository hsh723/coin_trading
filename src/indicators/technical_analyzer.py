"""
기술적 분석 모듈
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from functools import lru_cache
import time

class TechnicalAnalysisError(Exception):
    """기술적 분석 관련 예외"""
    pass

class DataNotFoundError(TechnicalAnalysisError):
    """데이터를 찾을 수 없는 경우"""
    pass

class CalculationError(TechnicalAnalysisError):
    """계산 중 오류가 발생한 경우"""
    pass

class TechnicalAnalyzer:
    """기술적 분석 클래스"""
    
    def __init__(self, db_manager):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        self._cache_timeout = 300  # 5분
        
    def _check_cache(self, key: str) -> Optional[Dict]:
        """
        캐시 확인
        
        Args:
            key (str): 캐시 키
            
        Returns:
            Optional[Dict]: 캐시된 데이터
        """
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_timeout:
                return data
            del self._cache[key]
        return None
        
    def _update_cache(self, key: str, data: Dict):
        """
        캐시 업데이트
        
        Args:
            key (str): 캐시 키
            data (Dict): 캐시할 데이터
        """
        self._cache[key] = (data, time.time())
        
    def calculate_indicators(self,
                           symbol: str,
                           timeframe: str,
                           indicators: List[str]) -> Dict:
        """
        기술적 지표 계산
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            indicators (List[str]): 계산할 지표 목록
            
        Returns:
            Dict: 계산된 지표
            
        Raises:
            DataNotFoundError: 가격 데이터를 찾을 수 없는 경우
            CalculationError: 지표 계산 중 오류가 발생한 경우
        """
        try:
            # 캐시 확인
            cache_key = f"{symbol}_{timeframe}_{'_'.join(sorted(indicators))}"
            cached_data = self._check_cache(cache_key)
            if cached_data:
                return cached_data
                
            # 가격 데이터 조회
            price_data = self.db_manager.get_price_data(symbol, timeframe)
            
            if price_data.empty:
                raise DataNotFoundError(f"{symbol}의 가격 데이터가 없습니다")
                
            # 지표 계산
            results = {}
            
            for indicator in indicators:
                try:
                    if indicator == 'sma':
                        results['sma_20'] = self._calculate_sma(price_data['close'], 20)
                        results['sma_50'] = self._calculate_sma(price_data['close'], 50)
                        results['sma_200'] = self._calculate_sma(price_data['close'], 200)
                        
                    elif indicator == 'ema':
                        results['ema_12'] = self._calculate_ema(price_data['close'], 12)
                        results['ema_26'] = self._calculate_ema(price_data['close'], 26)
                        
                    elif indicator == 'macd':
                        macd, signal, hist = self._calculate_macd(price_data['close'])
                        results['macd'] = macd
                        results['macd_signal'] = signal
                        results['macd_hist'] = hist
                        
                    elif indicator == 'rsi':
                        results['rsi'] = self._calculate_rsi(price_data['close'])
                        
                    elif indicator == 'bollinger':
                        upper, middle, lower = self._calculate_bollinger_bands(price_data['close'])
                        results['bb_upper'] = upper
                        results['bb_middle'] = middle
                        results['bb_lower'] = lower
                        
                    elif indicator == 'stochastic':
                        k, d = self._calculate_stochastic(price_data['high'], price_data['low'], price_data['close'])
                        results['stoch_k'] = k
                        results['stoch_d'] = d
                        
                    elif indicator == 'volume':
                        results['volume_sma'] = self._calculate_sma(price_data['volume'], 20)
                        
                except Exception as e:
                    self.logger.error(f"{indicator} 계산 중 오류 발생: {str(e)}")
                    raise CalculationError(f"{indicator} 계산 실패: {str(e)}")
                    
            # 결과 캐시
            self._update_cache(cache_key, results)
            
            return results
            
        except DataNotFoundError:
            raise
        except CalculationError:
            raise
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 실패: {str(e)}")
            raise TechnicalAnalysisError(f"기술적 지표 계산 실패: {str(e)}")
            
    @lru_cache(maxsize=100)
    def _calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """
        단순 이동평균 계산
        
        Args:
            data (pd.Series): 가격 데이터
            period (int): 기간
            
        Returns:
            pd.Series: 이동평균
        """
        return talib.SMA(data, timeperiod=period)
        
    @lru_cache(maxsize=100)
    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        지수 이동평균 계산
        
        Args:
            data (pd.Series): 가격 데이터
            period (int): 기간
            
        Returns:
            pd.Series: 지수 이동평균
        """
        return talib.EMA(data, timeperiod=period)
        
    @lru_cache(maxsize=100)
    def _calculate_macd(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD 계산
        
        Args:
            data (pd.Series): 가격 데이터
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD, 시그널, 히스토그램
        """
        macd, signal, hist = talib.MACD(data)
        return macd, signal, hist
        
    @lru_cache(maxsize=100)
    def _calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI 계산
        
        Args:
            data (pd.Series): 가격 데이터
            period (int): 기간
            
        Returns:
            pd.Series: RSI
        """
        return talib.RSI(data, timeperiod=period)
        
    @lru_cache(maxsize=100)
    def _calculate_bollinger_bands(self,
                                 data: pd.Series,
                                 period: int = 20,
                                 std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        볼린저 밴드 계산
        
        Args:
            data (pd.Series): 가격 데이터
            period (int): 기간
            std_dev (float): 표준편차 승수
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: 상단, 중간, 하단 밴드
        """
        upper, middle, lower = talib.BBANDS(data, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
        return upper, middle, lower
        
    @lru_cache(maxsize=100)
    def _calculate_stochastic(self,
                            high: pd.Series,
                            low: pd.Series,
                            close: pd.Series,
                            k_period: int = 14,
                            d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        스토캐스틱 계산
        
        Args:
            high (pd.Series): 고가
            low (pd.Series): 저가
            close (pd.Series): 종가
            k_period (int): %K 기간
            d_period (int): %D 기간
            
        Returns:
            Tuple[pd.Series, pd.Series]: %K, %D
        """
        k, d = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period)
        return k, d
        
    def get_support_resistance(self,
                             symbol: str,
                             timeframe: str,
                             lookback: int = 100) -> Dict:
        """
        지지/저항선 계산
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            lookback (int): 조회 기간
            
        Returns:
            Dict: 지지/저항선
            
        Raises:
            DataNotFoundError: 가격 데이터를 찾을 수 없는 경우
            CalculationError: 계산 중 오류가 발생한 경우
        """
        try:
            # 캐시 확인
            cache_key = f"sr_{symbol}_{timeframe}_{lookback}"
            cached_data = self._check_cache(cache_key)
            if cached_data:
                return cached_data
                
            # 가격 데이터 조회
            price_data = self.db_manager.get_price_data(symbol, timeframe)
            
            if price_data.empty:
                raise DataNotFoundError(f"{symbol}의 가격 데이터가 없습니다")
                
            # 최근 데이터만 사용
            recent_data = price_data.tail(lookback)
            
            # 지지/저항선 계산
            support_levels = self._find_support_levels(recent_data)
            resistance_levels = self._find_resistance_levels(recent_data)
            
            result = {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
            # 결과 캐시
            self._update_cache(cache_key, result)
            
            return result
            
        except DataNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"지지/저항선 계산 실패: {str(e)}")
            raise CalculationError(f"지지/저항선 계산 실패: {str(e)}")
            
    def _find_support_levels(self, data: pd.DataFrame, threshold: float = 0.02) -> List[float]:
        """
        지지선 찾기
        
        Args:
            data (pd.DataFrame): 가격 데이터
            threshold (float): 임계값
            
        Returns:
            List[float]: 지지선 목록
        """
        try:
            # 로컬 미니마 찾기
            local_minima = []
            
            for i in range(1, len(data) - 1):
                if data['low'].iloc[i] < data['low'].iloc[i-1] and data['low'].iloc[i] < data['low'].iloc[i+1]:
                    local_minima.append(data['low'].iloc[i])
                    
            # 임계값 기반 클러스터링
            support_levels = []
            current_cluster = []
            
            for price in sorted(local_minima):
                if not current_cluster or price - current_cluster[0] <= threshold * current_cluster[0]:
                    current_cluster.append(price)
                else:
                    if len(current_cluster) >= 3:  # 최소 3개의 포인트가 있어야 지지선으로 인정
                        support_levels.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [price]
                    
            if len(current_cluster) >= 3:
                support_levels.append(sum(current_cluster) / len(current_cluster))
                
            return sorted(support_levels)
            
        except Exception as e:
            self.logger.error(f"지지선 찾기 실패: {str(e)}")
            return []
            
    def _find_resistance_levels(self, data: pd.DataFrame, threshold: float = 0.02) -> List[float]:
        """
        저항선 찾기
        
        Args:
            data (pd.DataFrame): 가격 데이터
            threshold (float): 임계값
            
        Returns:
            List[float]: 저항선 목록
        """
        try:
            # 로컬 맥시마 찾기
            local_maxima = []
            
            for i in range(1, len(data) - 1):
                if data['high'].iloc[i] > data['high'].iloc[i-1] and data['high'].iloc[i] > data['high'].iloc[i+1]:
                    local_maxima.append(data['high'].iloc[i])
                    
            # 임계값 기반 클러스터링
            resistance_levels = []
            current_cluster = []
            
            for price in sorted(local_maxima):
                if not current_cluster or price - current_cluster[0] <= threshold * current_cluster[0]:
                    current_cluster.append(price)
                else:
                    if len(current_cluster) >= 3:  # 최소 3개의 포인트가 있어야 저항선으로 인정
                        resistance_levels.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [price]
                    
            if len(current_cluster) >= 3:
                resistance_levels.append(sum(current_cluster) / len(current_cluster))
                
            return sorted(resistance_levels)
            
        except Exception as e:
            self.logger.error(f"저항선 찾기 실패: {str(e)}")
            return []
            
    def get_trend(self,
                 symbol: str,
                 timeframe: str,
                 lookback: int = 100) -> Dict:
        """
        추세 분석
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            lookback (int): 조회 기간
            
        Returns:
            Dict: 추세 분석 결과
            
        Raises:
            DataNotFoundError: 가격 데이터를 찾을 수 없는 경우
            CalculationError: 계산 중 오류가 발생한 경우
        """
        try:
            # 캐시 확인
            cache_key = f"trend_{symbol}_{timeframe}_{lookback}"
            cached_data = self._check_cache(cache_key)
            if cached_data:
                return cached_data
                
            # 가격 데이터 조회
            price_data = self.db_manager.get_price_data(symbol, timeframe)
            
            if price_data.empty:
                raise DataNotFoundError(f"{symbol}의 가격 데이터가 없습니다")
                
            # 최근 데이터만 사용
            recent_data = price_data.tail(lookback)
            
            # 이동평균 계산
            sma_20 = self._calculate_sma(recent_data['close'], 20)
            sma_50 = self._calculate_sma(recent_data['close'], 50)
            
            # 현재 가격과 이동평균 비교
            current_price = recent_data['close'].iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            
            # 추세 판단
            trend = 'neutral'
            strength = 0
            
            if current_price > current_sma_20 > current_sma_50:
                trend = 'uptrend'
                strength = 1
            elif current_price < current_sma_20 < current_sma_50:
                trend = 'downtrend'
                strength = 1
            elif current_price > current_sma_50:
                trend = 'uptrend'
                strength = 0.5
            elif current_price < current_sma_50:
                trend = 'downtrend'
                strength = 0.5
                
            result = {
                'trend': trend,
                'strength': strength,
                'current_price': current_price,
                'sma_20': current_sma_20,
                'sma_50': current_sma_50
            }
            
            # 결과 캐시
            self._update_cache(cache_key, result)
            
            return result
            
        except DataNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"추세 분석 실패: {str(e)}")
            raise CalculationError(f"추세 분석 실패: {str(e)}") 