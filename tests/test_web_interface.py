import pytest
from src.web.web_server import WebServer
import os
import json
import requests
import time
import pandas as pd
import numpy as np

@pytest.fixture
def web_server():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "host": "localhost",
            "port": 8080,
            "api_endpoints": {
                "trading": "/api/trading",
                "analysis": "/api/analysis",
                "monitoring": "/api/monitoring",
                "settings": "/api/settings"
            },
            "authentication": {
                "enabled": True,
                "jwt_secret": "test_secret"
            }
        }
    }
    with open(os.path.join(config_dir, "web_server.json"), "w") as f:
        json.dump(config, f)
    
    return WebServer(config_dir=config_dir, data_dir=data_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="1D")
    data = pd.DataFrame({
        "open": np.random.normal(50000, 1000, len(dates)),
        "high": np.random.normal(51000, 1000, len(dates)),
        "low": np.random.normal(49000, 1000, len(dates)),
        "close": np.random.normal(50500, 1000, len(dates)),
        "volume": np.random.normal(100, 10, len(dates))
    }, index=dates)
    return data

def test_web_server_initialization(web_server):
    assert web_server is not None
    assert web_server.config_dir == "./config"
    assert web_server.data_dir == "./data"

def test_web_server_start_stop(web_server):
    web_server.start()
    assert web_server.is_running() is True
    
    web_server.stop()
    assert web_server.is_running() is False

def test_api_endpoints(web_server):
    web_server.start()
    
    # API 엔드포인트 확인
    endpoints = web_server.get_api_endpoints()
    assert endpoints is not None
    assert "/api/trading" in endpoints
    assert "/api/analysis" in endpoints
    assert "/api/monitoring" in endpoints
    assert "/api/settings" in endpoints
    
    web_server.stop()

def test_authentication(web_server):
    web_server.start()
    
    # 인증 토큰 생성
    token = web_server.generate_auth_token("test_user")
    assert token is not None
    
    # 토큰 검증
    is_valid = web_server.validate_auth_token(token)
    assert is_valid is True
    
    web_server.stop()

def test_trading_api(web_server, sample_data):
    web_server.start()
    
    # 거래 API 테스트
    response = web_server.handle_trading_request({
        "action": "place_order",
        "symbol": "BTCUSDT",
        "order_type": "limit",
        "side": "buy",
        "quantity": 0.1,
        "price": 50000.0
    })
    
    assert response is not None
    assert "status" in response
    assert "order_id" in response
    
    web_server.stop()

def test_analysis_api(web_server, sample_data):
    web_server.start()
    
    # 분석 API 테스트
    response = web_server.handle_analysis_request({
        "action": "analyze_performance",
        "data": sample_data.to_dict(),
        "start_date": "2023-01-01",
        "end_date": "2023-01-31"
    })
    
    assert response is not None
    assert "performance_metrics" in response
    assert "risk_metrics" in response
    
    web_server.stop()

def test_monitoring_api(web_server):
    web_server.start()
    
    # 모니터링 API 테스트
    response = web_server.handle_monitoring_request({
        "action": "get_system_status"
    })
    
    assert response is not None
    assert "status" in response
    assert "metrics" in response
    
    web_server.stop()

def test_settings_api(web_server):
    web_server.start()
    
    # 설정 API 테스트
    response = web_server.handle_settings_request({
        "action": "update_settings",
        "settings": {
            "risk_limits": {
                "max_position_size": 1.0,
                "max_leverage": 20
            }
        }
    })
    
    assert response is not None
    assert "status" in response
    assert "updated_settings" in response
    
    web_server.stop()

def test_web_socket(web_server):
    web_server.start()
    
    # 웹소켓 연결 테스트
    connection = web_server.create_websocket_connection()
    assert connection is not None
    
    # 메시지 전송 테스트
    message = {
        "type": "subscribe",
        "channel": "trades",
        "symbol": "BTCUSDT"
    }
    
    response = web_server.send_websocket_message(connection, message)
    assert response is not None
    assert "status" in response
    
    web_server.stop()

def test_error_handling(web_server):
    web_server.start()
    
    # 잘못된 API 요청 테스트
    with pytest.raises(Exception):
        web_server.handle_trading_request({
            "action": "invalid_action"
        })
    
    # 에러 통계 확인
    error_stats = web_server.get_error_stats()
    assert error_stats is not None
    assert error_stats["error_count"] > 0
    
    web_server.stop() 