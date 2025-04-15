import pytest
from src.config.config_manager import ConfigManager
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def config_manager():
    config_dir = "./config"
    os.makedirs(config_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "trading": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "exchange": "binance",
                "api_key": "test_api_key",
                "api_secret": "test_api_secret"
            },
            "strategy": {
                "type": "moving_average",
                "parameters": {
                    "short_window": 10,
                    "long_window": 20
                }
            },
            "risk": {
                "max_position_size": 0.1,
                "max_drawdown": 0.05,
                "stop_loss": 0.02
            }
        }
    }
    with open(os.path.join(config_dir, "config.json"), "w") as f:
        json.dump(config, f)
    
    return ConfigManager(config_dir=config_dir)

def test_config_manager_initialization(config_manager):
    assert config_manager is not None
    assert config_manager.config_dir == "./config"

def test_config_loading(config_manager):
    # 설정 파일 로딩 테스트
    config = config_manager.load_config()
    assert config is not None
    assert "default" in config
    assert "trading" in config["default"]
    assert "strategy" in config["default"]
    assert "risk" in config["default"]

def test_config_saving(config_manager):
    # 설정 파일 저장 테스트
    new_config = {
        "default": {
            "trading": {
                "symbols": ["BTCUSDT"],
                "timeframes": ["1m", "5m"],
                "exchange": "binance"
            }
        }
    }
    
    result = config_manager.save_config(new_config)
    assert result is True
    
    # 저장된 설정 확인
    loaded_config = config_manager.load_config()
    assert loaded_config["default"]["trading"]["symbols"] == ["BTCUSDT"]
    assert loaded_config["default"]["trading"]["timeframes"] == ["1m", "5m"]

def test_config_getting(config_manager):
    # 설정 값 조회 테스트
    # 전체 설정 조회
    config = config_manager.get_config()
    assert config is not None
    
    # 특정 섹션 조회
    trading_config = config_manager.get_config("trading")
    assert trading_config is not None
    assert "symbols" in trading_config
    assert "timeframes" in trading_config
    
    # 특정 키 조회
    symbols = config_manager.get_config("trading.symbols")
    assert symbols == ["BTCUSDT", "ETHUSDT"]

def test_config_setting(config_manager):
    # 설정 값 설정 테스트
    # 특정 섹션 설정
    new_trading_config = {
        "symbols": ["BTCUSDT"],
        "timeframes": ["1m"]
    }
    result = config_manager.set_config("trading", new_trading_config)
    assert result is True
    
    # 특정 키 설정
    result = config_manager.set_config("trading.symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    assert result is True
    
    # 설정 확인
    symbols = config_manager.get_config("trading.symbols")
    assert symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

def test_config_validation(config_manager):
    # 설정 값 검증 테스트
    # 유효한 설정
    valid_config = {
        "trading": {
            "symbols": ["BTCUSDT"],
            "timeframes": ["1m"]
        }
    }
    assert config_manager.validate_config(valid_config) is True
    
    # 잘못된 설정
    invalid_config = {
        "trading": {
            "symbols": "BTCUSDT",  # 리스트가 아닌 문자열
            "timeframes": ["1m"]
        }
    }
    assert config_manager.validate_config(invalid_config) is False

def test_config_error_handling(config_manager):
    # 설정 에러 처리 테스트
    # 존재하지 않는 설정 파일
    with pytest.raises(FileNotFoundError):
        config_manager.load_config("nonexistent.json")
    
    # 잘못된 JSON 형식
    invalid_json = "{invalid json}"
    with open(os.path.join(config_manager.config_dir, "invalid.json"), "w") as f:
        f.write(invalid_json)
    
    with pytest.raises(json.JSONDecodeError):
        config_manager.load_config("invalid.json")

def test_config_performance(config_manager):
    # 설정 성능 테스트
    # 대량의 설정 조회
    start_time = datetime.now()
    
    for i in range(1000):
        config_manager.get_config("trading.symbols")
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 1000개의 설정 조회를 1초 이내에 처리
    assert processing_time < 1.0

def test_config_backup(config_manager):
    # 설정 백업 테스트
    # 백업 생성
    result = config_manager.create_backup()
    assert result is True
    
    # 백업 파일 확인
    backup_files = [f for f in os.listdir(config_manager.config_dir) if f.endswith(".backup")]
    assert len(backup_files) > 0
    
    # 백업 복원
    result = config_manager.restore_backup(backup_files[0])
    assert result is True

def test_config_environment(config_manager):
    # 환경 변수 설정 테스트
    # 환경 변수 설정
    os.environ["TRADING_SYMBOLS"] = "BTCUSDT,ETHUSDT"
    
    # 환경 변수 기반 설정 로드
    config = config_manager.load_config_from_env()
    assert config is not None
    assert "trading" in config
    assert "symbols" in config["trading"]
    assert config["trading"]["symbols"] == ["BTCUSDT", "ETHUSDT"] 