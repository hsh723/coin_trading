"""
데이터 수집 모듈 테스트
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.data.collector import DataCollector

@pytest.fixture
def mock_exchange():
    """거래소 API 모의 객체 생성"""
    mock = MagicMock()
    mock.fetch_ohlcv.return_value = [
        [1625097600000, 35000, 36000, 34000, 35500, 100],
        [1625097900000, 35500, 36500, 35000, 36000, 120],
        [1625098200000, 36000, 37000, 35500, 36500, 150]
    ]
    return mock

@pytest.fixture
def collector(mock_exchange):
    """데이터 수집기 인스턴스 생성"""
    with patch('ccxt.binance', return_value=mock_exchange):
        collector = DataCollector(
            exchange='binance',
            symbols=['BTC/USDT'],
            timeframes=['1h']
        )
        return collector

def test_init(collector):
    """초기화 테스트"""
    assert collector.exchange is not None
    assert collector.symbols == ['BTC/USDT']
    assert collector.timeframes == ['1h']

def test_get_historical_data(collector):
    """히스토리컬 데이터 수집 테스트"""
    start_date = datetime(2021, 7, 1)
    end_date = datetime(2021, 7, 2)
    
    data = collector.get_historical_data(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date=start_date,
        end_date=end_date
    )
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 3
    assert all(col in data.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def test_get_historical_data_error(collector):
    """에러 처리 테스트"""
    collector.exchange.fetch_ohlcv.side_effect = Exception("API Error")
    
    with pytest.raises(Exception) as exc_info:
        collector.get_historical_data(
            symbol='BTC/USDT',
            timeframe='1h',
            start_date=datetime(2021, 7, 1),
            end_date=datetime(2021, 7, 2)
        )
    
    assert str(exc_info.value) == "API Error"

def test_get_historical_data_empty(collector):
    """빈 데이터 처리 테스트"""
    collector.exchange.fetch_ohlcv.return_value = []
    
    data = collector.get_historical_data(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date=datetime(2021, 7, 1),
        end_date=datetime(2021, 7, 2)
    )
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 0

def test_get_historical_data_invalid_dates(collector):
    """잘못된 날짜 처리 테스트"""
    with pytest.raises(ValueError):
        collector.get_historical_data(
            symbol='BTC/USDT',
            timeframe='1h',
            start_date=datetime(2021, 7, 2),
            end_date=datetime(2021, 7, 1)
        )

def test_collector_initialization(collector):
    """데이터 수집기 초기화 테스트"""
    assert collector.symbol == "BTC/USDT"
    assert collector.timeframe == "1h"
    assert collector.testnet is True
    assert collector.exchange is not None

def test_collect_historical_data(collector):
    """과거 데이터 수집 테스트"""
    # 최근 24시간 데이터 수집
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    data = collector.collect_historical_data(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    # 데이터 구조 검증
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert data.index.name == 'timestamp'
    
    # 데이터 타입 검증
    assert data['open'].dtype in ['float64', 'float32']
    assert data['high'].dtype in ['float64', 'float32']
    assert data['low'].dtype in ['float64', 'float32']
    assert data['close'].dtype in ['float64', 'float32']
    assert data['volume'].dtype in ['float64', 'float32']

def test_collect_realtime_data(collector):
    """실시간 데이터 수집 테스트"""
    data = collector.collect_realtime_data()
    
    # 데이터 구조 검증
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert data.index.name == 'timestamp'
    
    # 데이터 타입 검증
    assert data['open'].dtype in ['float64', 'float32']
    assert data['high'].dtype in ['float64', 'float32']
    assert data['low'].dtype in ['float64', 'float32']
    assert data['close'].dtype in ['float64', 'float32']
    assert data['volume'].dtype in ['float64', 'float32']

def test_invalid_symbol():
    """잘못된 심볼 테스트"""
    with pytest.raises(ValueError):
        DataCollector(
            symbol="INVALID/PAIR",
            timeframe="1h",
            testnet=True
        )

def test_invalid_timeframe():
    """잘못된 시간 프레임 테스트"""
    with pytest.raises(ValueError):
        DataCollector(
            symbol="BTC/USDT",
            timeframe="invalid",
            testnet=True
        )

def test_data_consistency(collector):
    """데이터 일관성 테스트"""
    # 과거 데이터 수집
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    data = collector.collect_historical_data(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    # OHLCV 데이터 검증
    assert (data['high'] >= data['open']).all()
    assert (data['high'] >= data['close']).all()
    assert (data['low'] <= data['open']).all()
    assert (data['low'] <= data['close']).all()
    assert (data['volume'] >= 0).all() 