import pytest
from src.cache.cache_manager import CacheManager
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def cache_manager():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "cache": {
                "type": "redis",
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "ttl": 3600,
                "max_size": 1000
            },
            "keys": {
                "price": "price:{symbol}:{timeframe}",
                "orderbook": "orderbook:{symbol}",
                "trades": "trades:{symbol}",
                "indicators": "indicators:{symbol}:{timeframe}"
            }
        }
    }
    with open(os.path.join(config_dir, "cache.json"), "w") as f:
        json.dump(config, f)
    
    return CacheManager(config_dir=config_dir, data_dir=data_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    n_samples = 100
    n_features = 10
    
    # 가격 데이터
    price_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='1H'),
        'open': np.random.normal(100, 10, n_samples),
        'high': np.random.normal(105, 10, n_samples),
        'low': np.random.normal(95, 10, n_samples),
        'close': np.random.normal(100, 10, n_samples),
        'volume': np.random.normal(1000, 100, n_samples)
    })
    
    # 호가 데이터
    orderbook_data = {
        'bids': [
            {'price': 99.0, 'quantity': 1.0},
            {'price': 98.0, 'quantity': 2.0},
            {'price': 97.0, 'quantity': 3.0}
        ],
        'asks': [
            {'price': 101.0, 'quantity': 1.0},
            {'price': 102.0, 'quantity': 2.0},
            {'price': 103.0, 'quantity': 3.0}
        ]
    }
    
    # 거래 데이터
    trades_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='1min'),
        'price': np.random.normal(100, 10, n_samples),
        'quantity': np.random.normal(1, 0.1, n_samples)
    })
    
    return price_data, orderbook_data, trades_data

def test_cache_manager_initialization(cache_manager):
    assert cache_manager is not None
    assert cache_manager.config_dir == "./config"
    assert cache_manager.data_dir == "./data"

def test_cache_set_get(cache_manager, sample_data):
    # 캐시 저장 및 조회 테스트
    price_data, orderbook_data, trades_data = sample_data
    
    # 가격 데이터 캐시
    cache_manager.set("price:BTCUSDT:1h", price_data)
    cached_price = cache_manager.get("price:BTCUSDT:1h")
    
    assert cached_price is not None
    assert isinstance(cached_price, pd.DataFrame)
    assert not cached_price.empty
    assert cached_price.equals(price_data)
    
    # 호가 데이터 캐시
    cache_manager.set("orderbook:BTCUSDT", orderbook_data)
    cached_orderbook = cache_manager.get("orderbook:BTCUSDT")
    
    assert cached_orderbook is not None
    assert cached_orderbook == orderbook_data
    
    # 거래 데이터 캐시
    cache_manager.set("trades:BTCUSDT", trades_data)
    cached_trades = cache_manager.get("trades:BTCUSDT")
    
    assert cached_trades is not None
    assert isinstance(cached_trades, pd.DataFrame)
    assert not cached_trades.empty
    assert cached_trades.equals(trades_data)

def test_cache_ttl(cache_manager, sample_data):
    # 캐시 TTL 테스트
    price_data, _, _ = sample_data
    
    # TTL이 1초인 캐시 설정
    cache_manager.set("price:BTCUSDT:1h", price_data, ttl=1)
    
    # 즉시 조회
    cached_data = cache_manager.get("price:BTCUSDT:1h")
    assert cached_data is not None
    
    # 2초 대기 후 조회
    import time
    time.sleep(2)
    expired_data = cache_manager.get("price:BTCUSDT:1h")
    assert expired_data is None

def test_cache_delete(cache_manager, sample_data):
    # 캐시 삭제 테스트
    price_data, _, _ = sample_data
    
    # 데이터 캐시
    cache_manager.set("price:BTCUSDT:1h", price_data)
    
    # 캐시 삭제
    cache_manager.delete("price:BTCUSDT:1h")
    
    # 삭제 확인
    deleted_data = cache_manager.get("price:BTCUSDT:1h")
    assert deleted_data is None

def test_cache_clear(cache_manager, sample_data):
    # 캐시 전체 삭제 테스트
    price_data, orderbook_data, trades_data = sample_data
    
    # 여러 데이터 캐시
    cache_manager.set("price:BTCUSDT:1h", price_data)
    cache_manager.set("orderbook:BTCUSDT", orderbook_data)
    cache_manager.set("trades:BTCUSDT", trades_data)
    
    # 캐시 전체 삭제
    cache_manager.clear()
    
    # 삭제 확인
    assert cache_manager.get("price:BTCUSDT:1h") is None
    assert cache_manager.get("orderbook:BTCUSDT") is None
    assert cache_manager.get("trades:BTCUSDT") is None

def test_cache_exists(cache_manager, sample_data):
    # 캐시 존재 여부 테스트
    price_data, _, _ = sample_data
    
    # 데이터 캐시
    cache_manager.set("price:BTCUSDT:1h", price_data)
    
    # 존재 확인
    assert cache_manager.exists("price:BTCUSDT:1h") is True
    assert cache_manager.exists("price:ETHUSDT:1h") is False

def test_cache_size(cache_manager, sample_data):
    # 캐시 크기 테스트
    price_data, orderbook_data, trades_data = sample_data
    
    # 여러 데이터 캐시
    cache_manager.set("price:BTCUSDT:1h", price_data)
    cache_manager.set("orderbook:BTCUSDT", orderbook_data)
    cache_manager.set("trades:BTCUSDT", trades_data)
    
    # 크기 확인
    size = cache_manager.size()
    assert size > 0

def test_cache_performance(cache_manager, sample_data):
    # 캐시 성능 테스트
    price_data, _, _ = sample_data
    
    # 대량의 데이터 캐시
    start_time = datetime.now()
    
    for i in range(100):
        key = f"price:BTCUSDT:1h:{i}"
        cache_manager.set(key, price_data)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 100개의 데이터를 1초 이내에 캐시
    assert processing_time < 1.0

def test_error_handling(cache_manager):
    # 에러 처리 테스트
    # 잘못된 키로 조회 시도
    with pytest.raises(Exception):
        cache_manager.get("invalid_key")

def test_cache_configuration(cache_manager):
    # 캐시 설정 테스트
    config = cache_manager.get_configuration()
    
    assert config is not None
    assert "cache" in config
    assert "keys" in config
    assert "type" in config["cache"]
    assert "host" in config["cache"]
    assert "port" in config["cache"]
    assert "db" in config["cache"]
    assert "ttl" in config["cache"]
    assert "max_size" in config["cache"] 