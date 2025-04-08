"""
기술적 분석기 테스트 모듈
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.indicators.technical_analyzer import TechnicalAnalyzer, TechnicalAnalysisError, DataNotFoundError, CalculationError

class MockDatabaseManager:
    """모의 데이터베이스 관리자"""
    
    def __init__(self):
        self.price_data = {}
        
    def get_price_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """가격 데이터 조회"""
        key = f"{symbol}_{timeframe}"
        if key not in self.price_data:
            return pd.DataFrame()
        return self.price_data[key]
        
    def set_price_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """가격 데이터 설정"""
        key = f"{symbol}_{timeframe}"
        self.price_data[key] = data

class TestTechnicalAnalyzer(unittest.TestCase):
    """기술적 분석기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.db_manager = MockDatabaseManager()
        self.analyzer = TechnicalAnalyzer(self.db_manager)
        
        # 테스트용 가격 데이터 생성
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = {
            'open': np.random.normal(100, 10, 100),
            'high': np.random.normal(105, 10, 100),
            'low': np.random.normal(95, 10, 100),
            'close': np.random.normal(100, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }
        
        df = pd.DataFrame(data, index=dates)
        self.db_manager.set_price_data('BTCUSDT', '1d', df)
        
    def test_calculate_indicators(self):
        """지표 계산 테스트"""
        # 정상 케이스
        indicators = ['sma', 'ema', 'macd', 'rsi', 'bollinger', 'stochastic', 'volume']
        result = self.analyzer.calculate_indicators('BTCUSDT', '1d', indicators)
        
        self.assertIn('sma_20', result)
        self.assertIn('sma_50', result)
        self.assertIn('sma_200', result)
        self.assertIn('ema_12', result)
        self.assertIn('ema_26', result)
        self.assertIn('macd', result)
        self.assertIn('macd_signal', result)
        self.assertIn('macd_hist', result)
        self.assertIn('rsi', result)
        self.assertIn('bb_upper', result)
        self.assertIn('bb_middle', result)
        self.assertIn('bb_lower', result)
        self.assertIn('stoch_k', result)
        self.assertIn('stoch_d', result)
        self.assertIn('volume_sma', result)
        
        # 데이터 없음 케이스
        with self.assertRaises(DataNotFoundError):
            self.analyzer.calculate_indicators('ETHUSDT', '1d', ['sma'])
            
        # 잘못된 지표 케이스
        with self.assertRaises(CalculationError):
            self.analyzer.calculate_indicators('BTCUSDT', '1d', ['invalid'])
            
    def test_get_support_resistance(self):
        """지지/저항선 테스트"""
        # 정상 케이스
        result = self.analyzer.get_support_resistance('BTCUSDT', '1d')
        
        self.assertIn('support', result)
        self.assertIn('resistance', result)
        self.assertIsInstance(result['support'], list)
        self.assertIsInstance(result['resistance'], list)
        
        # 데이터 없음 케이스
        with self.assertRaises(DataNotFoundError):
            self.analyzer.get_support_resistance('ETHUSDT', '1d')
            
    def test_get_trend(self):
        """추세 분석 테스트"""
        # 정상 케이스
        result = self.analyzer.get_trend('BTCUSDT', '1d')
        
        self.assertIn('trend', result)
        self.assertIn('strength', result)
        self.assertIn('current_price', result)
        self.assertIn('sma_20', result)
        self.assertIn('sma_50', result)
        
        self.assertIn(result['trend'], ['uptrend', 'downtrend', 'neutral'])
        self.assertIn(result['strength'], [0, 0.5, 1])
        
        # 데이터 없음 케이스
        with self.assertRaises(DataNotFoundError):
            self.analyzer.get_trend('ETHUSDT', '1d')
            
    def test_cache(self):
        """캐시 테스트"""
        # 첫 번째 호출
        start_time = datetime.now()
        result1 = self.analyzer.calculate_indicators('BTCUSDT', '1d', ['sma'])
        time1 = (datetime.now() - start_time).total_seconds()
        
        # 두 번째 호출 (캐시 사용)
        start_time = datetime.now()
        result2 = self.analyzer.calculate_indicators('BTCUSDT', '1d', ['sma'])
        time2 = (datetime.now() - start_time).total_seconds()
        
        # 결과가 동일한지 확인
        self.assertEqual(result1, result2)
        
        # 두 번째 호출이 더 빠른지 확인
        self.assertLess(time2, time1)
        
    def test_lru_cache(self):
        """LRU 캐시 테스트"""
        # 첫 번째 호출
        data = pd.Series([1, 2, 3, 4, 5])
        result1 = self.analyzer._calculate_sma(data, 3)
        
        # 두 번째 호출 (캐시 사용)
        result2 = self.analyzer._calculate_sma(data, 3)
        
        # 결과가 동일한지 확인
        self.assertTrue(result1.equals(result2))
        
if __name__ == '__main__':
    unittest.main() 