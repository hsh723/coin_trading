import pytest
from src.data.data_collector import DataCollector
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def data_collector():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "exchange": "binance",
            "api_key": "test_api_key",
            "api_secret": "test_api_secret",
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "data_types": ["trades", "orderbook", "kline"],
            "storage": {
                "type": "local",
                "path": "./data/raw"
            }
        }
    }
    with open(os.path.join(config_dir, "data_collection.json"), "w") as f:
        json.dump(config, f)
    
    return DataCollector(config_dir=config_dir, data_dir=data_dir)

def test_data_collector_initialization(data_collector):
    assert data_collector is not None
    assert data_collector.config_dir == "./config"
    assert data_collector.data_dir == "./data"

def test_historical_data_collection(data_collector):
    # 과거 데이터 수집 테스트
    symbol = "BTCUSDT"
    timeframe = "1h"
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()
    
    data = data_collector.collect_historical_data(symbol, timeframe, start_time, end_time)
    
    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "timestamp" in data.columns
    assert "open" in data.columns
    assert "high" in data.columns
    assert "low" in data.columns
    assert "close" in data.columns
    assert "volume" in data.columns

def test_realtime_data_collection(data_collector):
    # 실시간 데이터 수집 테스트
    symbol = "BTCUSDT"
    data_type = "trades"
    
    data = data_collector.collect_realtime_data(symbol, data_type)
    
    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "timestamp" in data.columns
    assert "price" in data.columns
    assert "quantity" in data.columns

def test_orderbook_collection(data_collector):
    # 호가 데이터 수집 테스트
    symbol = "BTCUSDT"
    
    orderbook = data_collector.collect_orderbook(symbol)
    
    assert orderbook is not None
    assert "bids" in orderbook
    assert "asks" in orderbook
    assert len(orderbook["bids"]) > 0
    assert len(orderbook["asks"]) > 0
    assert all("price" in bid for bid in orderbook["bids"])
    assert all("quantity" in bid for bid in orderbook["bids"])
    assert all("price" in ask for ask in orderbook["asks"])
    assert all("quantity" in ask for ask in orderbook["asks"])

def test_data_storage(data_collector):
    # 데이터 저장 테스트
    symbol = "BTCUSDT"
    timeframe = "1h"
    data = pd.DataFrame({
        "timestamp": [datetime.now()],
        "open": [100.0],
        "high": [105.0],
        "low": [95.0],
        "close": [100.0],
        "volume": [1000.0]
    })
    
    file_path = data_collector.store_data(symbol, timeframe, data)
    
    assert os.path.exists(file_path)
    os.remove(file_path)

def test_data_retrieval(data_collector):
    # 데이터 조회 테스트
    symbol = "BTCUSDT"
    timeframe = "1h"
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now()
    
    data = data_collector.retrieve_data(symbol, timeframe, start_time, end_time)
    
    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "timestamp" in data.columns
    assert "open" in data.columns
    assert "high" in data.columns
    assert "low" in data.columns
    assert "close" in data.columns
    assert "volume" in data.columns

def test_data_validation(data_collector):
    # 데이터 검증 테스트
    symbol = "BTCUSDT"
    timeframe = "1h"
    data = pd.DataFrame({
        "timestamp": [datetime.now()],
        "open": [100.0],
        "high": [105.0],
        "low": [95.0],
        "close": [100.0],
        "volume": [1000.0]
    })
    
    is_valid = data_collector.validate_data(symbol, timeframe, data)
    
    assert is_valid

def test_data_cleaning(data_collector):
    # 데이터 정제 테스트
    symbol = "BTCUSDT"
    timeframe = "1h"
    data = pd.DataFrame({
        "timestamp": [datetime.now()],
        "open": [100.0],
        "high": [105.0],
        "low": [95.0],
        "close": [100.0],
        "volume": [1000.0]
    })
    
    cleaned_data = data_collector.clean_data(symbol, timeframe, data)
    
    assert cleaned_data is not None
    assert isinstance(cleaned_data, pd.DataFrame)
    assert not cleaned_data.empty
    assert not cleaned_data.isna().any().any()

def test_data_aggregation(data_collector):
    # 데이터 집계 테스트
    symbol = "BTCUSDT"
    from_timeframe = "1m"
    to_timeframe = "1h"
    data = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=60, freq="1min"),
        "open": np.random.normal(100, 10, 60),
        "high": np.random.normal(105, 10, 60),
        "low": np.random.normal(95, 10, 60),
        "close": np.random.normal(100, 10, 60),
        "volume": np.random.normal(1000, 100, 60)
    })
    
    aggregated_data = data_collector.aggregate_data(symbol, from_timeframe, to_timeframe, data)
    
    assert aggregated_data is not None
    assert isinstance(aggregated_data, pd.DataFrame)
    assert not aggregated_data.empty
    assert len(aggregated_data) == 1  # 60분 데이터를 1시간으로 집계

def test_collection_performance(data_collector):
    # 수집 성능 테스트
    symbol = "BTCUSDT"
    timeframe = "1h"
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()
    
    start = datetime.now()
    data_collector.collect_historical_data(symbol, timeframe, start_time, end_time)
    end = datetime.now()
    
    processing_time = (end - start).total_seconds()
    
    # 성능 기준: 7일치 데이터 수집을 10초 이내에 완료
    assert processing_time < 10.0

def test_error_handling(data_collector):
    # 에러 처리 테스트
    # 잘못된 심볼로 데이터 수집 시도
    invalid_symbol = "INVALID"
    timeframe = "1h"
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now()
    
    with pytest.raises(Exception):
        data_collector.collect_historical_data(invalid_symbol, timeframe, start_time, end_time)

def test_collection_configuration(data_collector):
    # 수집 설정 테스트
    config = data_collector.get_configuration()
    
    assert config is not None
    assert "exchange" in config
    assert "api_key" in config
    assert "api_secret" in config
    assert "symbols" in config
    assert "timeframes" in config
    assert "data_types" in config
    assert "storage" in config 