import pytest
from src.websocket.websocket_manager import WebSocketManager
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import websockets
import time

@pytest.fixture
def websocket_manager():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "websocket": {
                "type": "binance",
                "url": "wss://stream.binance.com:9443/ws",
                "symbols": ["btcusdt", "ethusdt"],
                "channels": ["trade", "kline_1m", "depth"],
                "reconnect_interval": 5,
                "ping_interval": 30
            }
        }
    }
    with open(os.path.join(config_dir, "websocket.json"), "w") as f:
        json.dump(config, f)
    
    return WebSocketManager(config_dir=config_dir, data_dir=data_dir)

@pytest.fixture
def sample_messages():
    # 샘플 메시지 생성
    trade_message = {
        "e": "trade",
        "E": int(datetime.now().timestamp() * 1000),
        "s": "BTCUSDT",
        "t": 123456789,
        "p": "50000.00",
        "q": "1.00000000",
        "b": 88,
        "a": 50,
        "T": int(datetime.now().timestamp() * 1000),
        "m": True,
        "M": True
    }
    
    kline_message = {
        "e": "kline",
        "E": int(datetime.now().timestamp() * 1000),
        "s": "BTCUSDT",
        "k": {
            "t": int(datetime.now().timestamp() * 1000),
            "T": int((datetime.now() + timedelta(minutes=1)).timestamp() * 1000),
            "s": "BTCUSDT",
            "i": "1m",
            "f": 100,
            "L": 200,
            "o": "50000.00",
            "c": "50100.00",
            "h": "50200.00",
            "l": "49900.00",
            "v": "100.00000000",
            "n": 100,
            "x": False,
            "q": "5000000.00000000",
            "V": "50.00000000",
            "Q": "2500000.00000000",
            "B": "0"
        }
    }
    
    depth_message = {
        "e": "depthUpdate",
        "E": int(datetime.now().timestamp() * 1000),
        "s": "BTCUSDT",
        "U": 157,
        "u": 160,
        "b": [
            ["50000.00", "1.00000000"],
            ["49900.00", "2.00000000"]
        ],
        "a": [
            ["50100.00", "1.00000000"],
            ["50200.00", "2.00000000"]
        ]
    }
    
    return trade_message, kline_message, depth_message

def test_websocket_manager_initialization(websocket_manager):
    assert websocket_manager is not None
    assert websocket_manager.config_dir == "./config"
    assert websocket_manager.data_dir == "./data"

def test_websocket_connection(websocket_manager):
    # WebSocket 연결 테스트
    result = websocket_manager.connect()
    assert result is True
    
    # 연결 상태 확인
    assert websocket_manager.is_connected() is True
    
    # 연결 종료
    websocket_manager.disconnect()
    assert websocket_manager.is_connected() is False

def test_websocket_subscription(websocket_manager):
    # WebSocket 구독 테스트
    websocket_manager.connect()
    
    # 심볼 구독
    result = websocket_manager.subscribe("btcusdt", ["trade", "kline_1m"])
    assert result is True
    
    # 구독 상태 확인
    subscriptions = websocket_manager.get_subscriptions()
    assert "btcusdt" in subscriptions
    assert "trade" in subscriptions["btcusdt"]
    assert "kline_1m" in subscriptions["btcusdt"]
    
    websocket_manager.disconnect()

def test_websocket_message_handling(websocket_manager, sample_messages):
    # WebSocket 메시지 처리 테스트
    trade_message, kline_message, depth_message = sample_messages
    
    websocket_manager.connect()
    websocket_manager.subscribe("btcusdt", ["trade", "kline_1m", "depth"])
    
    # 메시지 처리 함수 설정
    received_messages = []
    def message_handler(message):
        received_messages.append(message)
    
    websocket_manager.set_message_handler(message_handler)
    
    # 메시지 전송
    websocket_manager._handle_message(json.dumps(trade_message))
    websocket_manager._handle_message(json.dumps(kline_message))
    websocket_manager._handle_message(json.dumps(depth_message))
    
    # 메시지 수신 확인
    assert len(received_messages) == 3
    
    websocket_manager.disconnect()

def test_websocket_reconnection(websocket_manager):
    # WebSocket 재연결 테스트
    websocket_manager.connect()
    
    # 연결 강제 종료
    websocket_manager._connection.close()
    
    # 재연결 대기
    time.sleep(websocket_manager.config["websocket"]["reconnect_interval"] + 1)
    
    # 재연결 확인
    assert websocket_manager.is_connected() is True
    
    websocket_manager.disconnect()

def test_websocket_ping_pong(websocket_manager):
    # WebSocket 핑퐁 테스트
    websocket_manager.connect()
    
    # 핑 메시지 전송
    result = websocket_manager.ping()
    assert result is True
    
    # 퐁 응답 확인
    assert websocket_manager.last_pong is not None
    
    websocket_manager.disconnect()

def test_websocket_error_handling(websocket_manager):
    # WebSocket 에러 처리 테스트
    # 잘못된 URL로 연결 시도
    websocket_manager.config["websocket"]["url"] = "wss://invalid.url"
    
    with pytest.raises(Exception):
        websocket_manager.connect()

def test_websocket_performance(websocket_manager, sample_messages):
    # WebSocket 성능 테스트
    trade_message, _, _ = sample_messages
    
    websocket_manager.connect()
    websocket_manager.subscribe("btcusdt", ["trade"])
    
    # 대량의 메시지 처리
    start_time = datetime.now()
    
    for i in range(100):
        websocket_manager._handle_message(json.dumps(trade_message))
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 100개의 메시지를 1초 이내에 처리
    assert processing_time < 1.0
    
    websocket_manager.disconnect()

def test_websocket_configuration(websocket_manager):
    # WebSocket 설정 테스트
    config = websocket_manager.get_configuration()
    
    assert config is not None
    assert "websocket" in config
    assert "type" in config["websocket"]
    assert "url" in config["websocket"]
    assert "symbols" in config["websocket"]
    assert "channels" in config["websocket"]
    assert "reconnect_interval" in config["websocket"]
    assert "ping_interval" in config["websocket"] 