import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.data.collector import DataCollector

def test_data_collector_initialization():
    """데이터 수집기 초기화 테스트"""
    collector = DataCollector()
    assert collector is not None
    assert collector.exchange is not None

def test_get_historical_data(mock_binance_api):
    """과거 데이터 수집 테스트"""
    collector = DataCollector()
    collector.exchange = mock_binance_api
    
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()
    
    data = collector.get_historical_data(
        symbol='BTC/USDT',
        start_time=start_time,
        end_time=end_time,
        interval='1h'
    )
    
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert 'timestamp' in data.columns
    assert 'open' in data.columns
    assert 'high' in data.columns
    assert 'low' in data.columns
    assert 'close' in data.columns
    assert 'volume' in data.columns

def test_get_realtime_data(mock_binance_api):
    """실시간 데이터 수집 테스트"""
    collector = DataCollector()
    collector.exchange = mock_binance_api
    
    data = collector.get_realtime_data(symbol='BTC/USDT')
    
    assert isinstance(data, dict)
    assert 'symbol' in data
    assert 'last' in data
    assert 'bid' in data
    assert 'ask' in data
    assert 'volume' in data

def test_invalid_symbol():
    """잘못된 심볼로 데이터 수집 테스트"""
    collector = DataCollector()
    
    with pytest.raises(ValueError):
        collector.get_historical_data(
            symbol='INVALID/PAIR',
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            interval='1h'
        )

def test_invalid_time_range():
    """잘못된 시간 범위로 데이터 수집 테스트"""
    collector = DataCollector()
    
    with pytest.raises(ValueError):
        collector.get_historical_data(
            symbol='BTC/USDT',
            start_time=datetime.now(),
            end_time=datetime.now() - timedelta(days=1),
            interval='1h'
        )

def test_invalid_interval():
    """잘못된 간격으로 데이터 수집 테스트"""
    collector = DataCollector()
    
    with pytest.raises(ValueError):
        collector.get_historical_data(
            symbol='BTC/USDT',
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            interval='invalid'
        )

def test_data_consistency(mock_binance_api):
    """데이터 일관성 테스트"""
    collector = DataCollector()
    collector.exchange = mock_binance_api
    
    # 1시간 간격 데이터
    data_1h = collector.get_historical_data(
        symbol='BTC/USDT',
        start_time=datetime.now() - timedelta(days=1),
        end_time=datetime.now(),
        interval='1h'
    )
    
    # 4시간 간격 데이터
    data_4h = collector.get_historical_data(
        symbol='BTC/USDT',
        start_time=datetime.now() - timedelta(days=1),
        end_time=datetime.now(),
        interval='4h'
    )
    
    assert len(data_1h) >= len(data_4h)
    assert data_1h['close'].iloc[-1] == data_4h['close'].iloc[-1]

def test_data_limits(mock_binance_api):
    """데이터 제한 테스트"""
    collector = DataCollector()
    collector.exchange = mock_binance_api
    
    # 너무 긴 기간의 데이터 요청
    with pytest.raises(ValueError):
        collector.get_historical_data(
            symbol='BTC/USDT',
            start_time=datetime.now() - timedelta(days=365),
            end_time=datetime.now(),
            interval='1m'
        )
    
    # 너무 많은 데이터 요청
    with pytest.raises(ValueError):
        collector.get_historical_data(
            symbol='BTC/USDT',
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now(),
            interval='1m',
            limit=10000
        )

def test_data_collection():
    # Example test for data collection
    assert True  # Replace with actual test logic