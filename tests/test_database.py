import pytest
from src.database.database_manager import DatabaseManager
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3

@pytest.fixture
def database_manager():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "database": {
                "type": "sqlite",
                "path": "./data/trading.db",
                "tables": {
                    "prices": {
                        "columns": ["symbol", "price", "timestamp"],
                        "indexes": ["symbol", "timestamp"]
                    },
                    "trades": {
                        "columns": ["id", "symbol", "price", "quantity", "timestamp"],
                        "indexes": ["id", "symbol", "timestamp"]
                    },
                    "orders": {
                        "columns": ["id", "symbol", "side", "price", "quantity", "status", "timestamp"],
                        "indexes": ["id", "symbol", "status", "timestamp"]
                    }
                }
            }
        }
    }
    with open(os.path.join(config_dir, "database.json"), "w") as f:
        json.dump(config, f)
    
    return DatabaseManager(config_dir=config_dir, data_dir=data_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    prices = pd.DataFrame({
        "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        "price": [50000.0, 3000.0, 50100.0],
        "timestamp": [
            datetime.now() - timedelta(minutes=2),
            datetime.now() - timedelta(minutes=1),
            datetime.now()
        ]
    })
    
    trades = pd.DataFrame({
        "id": [1, 2, 3],
        "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        "price": [50000.0, 3000.0, 50100.0],
        "quantity": [1.0, 10.0, 0.5],
        "timestamp": [
            datetime.now() - timedelta(minutes=2),
            datetime.now() - timedelta(minutes=1),
            datetime.now()
        ]
    })
    
    orders = pd.DataFrame({
        "id": [1, 2, 3],
        "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        "side": ["BUY", "SELL", "BUY"],
        "price": [50000.0, 3000.0, 50100.0],
        "quantity": [1.0, 10.0, 0.5],
        "status": ["FILLED", "FILLED", "PENDING"],
        "timestamp": [
            datetime.now() - timedelta(minutes=2),
            datetime.now() - timedelta(minutes=1),
            datetime.now()
        ]
    })
    
    return prices, trades, orders

def test_database_manager_initialization(database_manager):
    assert database_manager is not None
    assert database_manager.config_dir == "./config"
    assert database_manager.data_dir == "./data"

def test_database_connection(database_manager):
    # 데이터베이스 연결 테스트
    result = database_manager.connect()
    assert result is True
    
    # 연결 상태 확인
    assert database_manager.is_connected() is True
    
    # 연결 종료
    database_manager.disconnect()
    assert database_manager.is_connected() is False

def test_database_tables(database_manager):
    # 데이터베이스 테이블 생성 테스트
    database_manager.connect()
    
    # 테이블 생성
    result = database_manager.create_tables()
    assert result is True
    
    # 테이블 존재 확인
    tables = database_manager.get_tables()
    assert "prices" in tables
    assert "trades" in tables
    assert "orders" in tables
    
    database_manager.disconnect()

def test_database_insert(database_manager, sample_data):
    # 데이터베이스 삽입 테스트
    prices, trades, orders = sample_data
    
    database_manager.connect()
    database_manager.create_tables()
    
    # 데이터 삽입
    result = database_manager.insert("prices", prices)
    assert result is True
    
    result = database_manager.insert("trades", trades)
    assert result is True
    
    result = database_manager.insert("orders", orders)
    assert result is True
    
    database_manager.disconnect()

def test_database_select(database_manager, sample_data):
    # 데이터베이스 조회 테스트
    prices, trades, orders = sample_data
    
    database_manager.connect()
    database_manager.create_tables()
    database_manager.insert("prices", prices)
    database_manager.insert("trades", trades)
    database_manager.insert("orders", orders)
    
    # 데이터 조회
    result = database_manager.select("prices", where="symbol = 'BTCUSDT'")
    assert len(result) == 2
    
    result = database_manager.select("trades", where="symbol = 'ETHUSDT'")
    assert len(result) == 1
    
    result = database_manager.select("orders", where="status = 'PENDING'")
    assert len(result) == 1
    
    database_manager.disconnect()

def test_database_update(database_manager, sample_data):
    # 데이터베이스 업데이트 테스트
    prices, trades, orders = sample_data
    
    database_manager.connect()
    database_manager.create_tables()
    database_manager.insert("prices", prices)
    database_manager.insert("trades", trades)
    database_manager.insert("orders", orders)
    
    # 데이터 업데이트
    result = database_manager.update("orders", 
                                   {"status": "FILLED"}, 
                                   "id = 3")
    assert result is True
    
    # 업데이트 확인
    result = database_manager.select("orders", where="id = 3")
    assert result.iloc[0]["status"] == "FILLED"
    
    database_manager.disconnect()

def test_database_delete(database_manager, sample_data):
    # 데이터베이스 삭제 테스트
    prices, trades, orders = sample_data
    
    database_manager.connect()
    database_manager.create_tables()
    database_manager.insert("prices", prices)
    database_manager.insert("trades", trades)
    database_manager.insert("orders", orders)
    
    # 데이터 삭제
    result = database_manager.delete("orders", "status = 'PENDING'")
    assert result is True
    
    # 삭제 확인
    result = database_manager.select("orders", where="status = 'PENDING'")
    assert len(result) == 0
    
    database_manager.disconnect()

def test_database_performance(database_manager, sample_data):
    # 데이터베이스 성능 테스트
    prices, trades, orders = sample_data
    
    database_manager.connect()
    database_manager.create_tables()
    
    # 대량의 데이터 삽입
    start_time = datetime.now()
    
    for i in range(100):
        database_manager.insert("prices", prices)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 100개의 데이터를 5초 이내에 삽입
    assert processing_time < 5.0
    
    database_manager.disconnect()

def test_database_error_handling(database_manager):
    # 데이터베이스 에러 처리 테스트
    database_manager.connect()
    
    # 잘못된 테이블 조회
    with pytest.raises(Exception):
        database_manager.select("invalid_table")
    
    # 잘못된 컬럼 삽입
    with pytest.raises(Exception):
        database_manager.insert("prices", pd.DataFrame({"invalid": [1, 2, 3]}))
    
    database_manager.disconnect()

def test_database_configuration(database_manager):
    # 데이터베이스 설정 테스트
    config = database_manager.get_configuration()
    
    assert config is not None
    assert "database" in config
    assert "type" in config["database"]
    assert "path" in config["database"]
    assert "tables" in config["database"]
    assert "prices" in config["database"]["tables"]
    assert "trades" in config["database"]["tables"]
    assert "orders" in config["database"]["tables"] 