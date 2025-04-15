import pytest
from src.api.api_manager import APIManager
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

@pytest.fixture
def api_manager():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "api": {
                "type": "binance",
                "base_url": "https://api.binance.com",
                "api_key": "test_api_key",
                "api_secret": "test_api_secret",
                "rate_limit": 1200,
                "rate_limit_period": 60
            }
        }
    }
    with open(os.path.join(config_dir, "api.json"), "w") as f:
        json.dump(config, f)
    
    return APIManager(config_dir=config_dir, data_dir=data_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    price_data = {
        "symbol": "BTCUSDT",
        "price": "50000.00",
        "time": int(datetime.now().timestamp() * 1000)
    }
    
    order_book = {
        "lastUpdateId": 1027024,
        "bids": [
            ["50000.00", "1.00000000"],
            ["49900.00", "2.00000000"]
        ],
        "asks": [
            ["50100.00", "1.00000000"],
            ["50200.00", "2.00000000"]
        ]
    }
    
    trades = [
        {
            "id": 28457,
            "price": "50000.00",
            "qty": "1.00000000",
            "time": int(datetime.now().timestamp() * 1000),
            "isBuyerMaker": True,
            "isBestMatch": True
        }
    ]
    
    return price_data, order_book, trades

def test_api_manager_initialization(api_manager):
    assert api_manager is not None
    assert api_manager.config_dir == "./config"
    assert api_manager.data_dir == "./data"

def test_api_authentication(api_manager):
    # API 인증 테스트
    result = api_manager.authenticate()
    assert result is True
    
    # 인증 상태 확인
    assert api_manager.is_authenticated() is True

def test_api_request(api_manager):
    # API 요청 테스트
    api_manager.authenticate()
    
    # 심볼 정보 요청
    response = api_manager.request("GET", "/api/v3/exchangeInfo")
    assert response is not None
    assert "symbols" in response

def test_api_rate_limit(api_manager):
    # API 요청 제한 테스트
    api_manager.authenticate()
    
    # 연속 요청
    start_time = datetime.now()
    for i in range(10):
        api_manager.request("GET", "/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 요청 제한 확인 (10개 요청이 1초 이내에 처리되어야 함)
    assert processing_time < 1.0

def test_api_error_handling(api_manager):
    # API 에러 처리 테스트
    api_manager.authenticate()
    
    # 잘못된 엔드포인트 요청
    with pytest.raises(Exception):
        api_manager.request("GET", "/api/v3/invalid_endpoint")
    
    # 잘못된 파라미터 요청
    with pytest.raises(Exception):
        api_manager.request("GET", "/api/v3/ticker/price", params={"invalid": "param"})

def test_api_data_processing(api_manager, sample_data):
    # API 데이터 처리 테스트
    price_data, order_book, trades = sample_data
    
    # 가격 데이터 처리
    processed_price = api_manager.process_price_data(price_data)
    assert processed_price is not None
    assert "symbol" in processed_price
    assert "price" in processed_price
    assert "timestamp" in processed_price
    
    # 호가 데이터 처리
    processed_order_book = api_manager.process_order_book_data(order_book)
    assert processed_order_book is not None
    assert "bids" in processed_order_book
    assert "asks" in processed_order_book
    
    # 거래 데이터 처리
    processed_trades = api_manager.process_trade_data(trades)
    assert processed_trades is not None
    assert len(processed_trades) > 0
    assert "id" in processed_trades[0]
    assert "price" in processed_trades[0]
    assert "quantity" in processed_trades[0]

def test_api_data_storage(api_manager, sample_data):
    # API 데이터 저장 테스트
    price_data, order_book, trades = sample_data
    
    # 데이터 저장
    result = api_manager.store_data("price", price_data)
    assert result is True
    
    result = api_manager.store_data("order_book", order_book)
    assert result is True
    
    result = api_manager.store_data("trades", trades)
    assert result is True
    
    # 데이터 조회
    stored_price = api_manager.retrieve_data("price")
    assert stored_price is not None
    
    stored_order_book = api_manager.retrieve_data("order_book")
    assert stored_order_book is not None
    
    stored_trades = api_manager.retrieve_data("trades")
    assert stored_trades is not None

def test_api_performance(api_manager):
    # API 성능 테스트
    api_manager.authenticate()
    
    # 대량의 요청 처리
    start_time = datetime.now()
    
    for i in range(100):
        api_manager.request("GET", "/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 100개의 요청을 5초 이내에 처리
    assert processing_time < 5.0

def test_api_configuration(api_manager):
    # API 설정 테스트
    config = api_manager.get_configuration()
    
    assert config is not None
    assert "api" in config
    assert "type" in config["api"]
    assert "base_url" in config["api"]
    assert "api_key" in config["api"]
    assert "api_secret" in config["api"]
    assert "rate_limit" in config["api"]
    assert "rate_limit_period" in config["api"] 