import pytest
from src.message.message_queue import MessageQueue
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def message_queue():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "queue": {
                "type": "rabbitmq",
                "host": "localhost",
                "port": 5672,
                "username": "guest",
                "password": "guest",
                "vhost": "/",
                "exchange": "trading",
                "queues": {
                    "trades": "trades",
                    "orders": "orders",
                    "signals": "signals",
                    "alerts": "alerts"
                }
            }
        }
    }
    with open(os.path.join(config_dir, "message_queue.json"), "w") as f:
        json.dump(config, f)
    
    return MessageQueue(config_dir=config_dir, data_dir=data_dir)

@pytest.fixture
def sample_messages():
    # 샘플 메시지 생성
    trade_message = {
        "type": "trade",
        "symbol": "BTCUSDT",
        "price": 50000.0,
        "quantity": 1.0,
        "timestamp": datetime.now().isoformat()
    }
    
    order_message = {
        "type": "order",
        "symbol": "BTCUSDT",
        "side": "buy",
        "price": 50000.0,
        "quantity": 1.0,
        "timestamp": datetime.now().isoformat()
    }
    
    signal_message = {
        "type": "signal",
        "symbol": "BTCUSDT",
        "signal": "buy",
        "strength": 0.8,
        "timestamp": datetime.now().isoformat()
    }
    
    alert_message = {
        "type": "alert",
        "level": "warning",
        "message": "Price volatility high",
        "timestamp": datetime.now().isoformat()
    }
    
    return trade_message, order_message, signal_message, alert_message

def test_message_queue_initialization(message_queue):
    assert message_queue is not None
    assert message_queue.config_dir == "./config"
    assert message_queue.data_dir == "./data"

def test_message_publish(message_queue, sample_messages):
    # 메시지 발행 테스트
    trade_message, order_message, signal_message, alert_message = sample_messages
    
    # 거래 메시지 발행
    result = message_queue.publish("trades", trade_message)
    assert result is True
    
    # 주문 메시지 발행
    result = message_queue.publish("orders", order_message)
    assert result is True
    
    # 신호 메시지 발행
    result = message_queue.publish("signals", signal_message)
    assert result is True
    
    # 알림 메시지 발행
    result = message_queue.publish("alerts", alert_message)
    assert result is True

def test_message_consume(message_queue, sample_messages):
    # 메시지 소비 테스트
    trade_message, order_message, signal_message, alert_message = sample_messages
    
    # 메시지 발행
    message_queue.publish("trades", trade_message)
    message_queue.publish("orders", order_message)
    message_queue.publish("signals", signal_message)
    message_queue.publish("alerts", alert_message)
    
    # 메시지 소비
    consumed_trade = message_queue.consume("trades")
    consumed_order = message_queue.consume("orders")
    consumed_signal = message_queue.consume("signals")
    consumed_alert = message_queue.consume("alerts")
    
    assert consumed_trade == trade_message
    assert consumed_order == order_message
    assert consumed_signal == signal_message
    assert consumed_alert == alert_message

def test_message_acknowledge(message_queue, sample_messages):
    # 메시지 확인 테스트
    trade_message, _, _, _ = sample_messages
    
    # 메시지 발행
    message_queue.publish("trades", trade_message)
    
    # 메시지 소비 및 확인
    message = message_queue.consume("trades")
    result = message_queue.acknowledge("trades", message)
    
    assert result is True

def test_message_reject(message_queue, sample_messages):
    # 메시지 거부 테스트
    trade_message, _, _, _ = sample_messages
    
    # 메시지 발행
    message_queue.publish("trades", trade_message)
    
    # 메시지 소비 및 거부
    message = message_queue.consume("trades")
    result = message_queue.reject("trades", message)
    
    assert result is True

def test_queue_status(message_queue):
    # 큐 상태 테스트
    status = message_queue.get_queue_status("trades")
    
    assert status is not None
    assert "message_count" in status
    assert "consumer_count" in status
    assert isinstance(status["message_count"], int)
    assert isinstance(status["consumer_count"], int)

def test_queue_purge(message_queue, sample_messages):
    # 큐 비우기 테스트
    trade_message, _, _, _ = sample_messages
    
    # 메시지 발행
    message_queue.publish("trades", trade_message)
    
    # 큐 비우기
    result = message_queue.purge("trades")
    assert result is True
    
    # 큐 상태 확인
    status = message_queue.get_queue_status("trades")
    assert status["message_count"] == 0

def test_message_performance(message_queue, sample_messages):
    # 메시지 처리 성능 테스트
    trade_message, _, _, _ = sample_messages
    
    # 대량의 메시지 발행
    start_time = datetime.now()
    
    for i in range(100):
        message_queue.publish("trades", trade_message)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 100개의 메시지를 1초 이내에 발행
    assert processing_time < 1.0

def test_error_handling(message_queue):
    # 에러 처리 테스트
    # 잘못된 큐 이름으로 메시지 발행 시도
    with pytest.raises(Exception):
        message_queue.publish("invalid_queue", {"test": "message"})

def test_message_configuration(message_queue):
    # 메시지 큐 설정 테스트
    config = message_queue.get_configuration()
    
    assert config is not None
    assert "queue" in config
    assert "type" in config["queue"]
    assert "host" in config["queue"]
    assert "port" in config["queue"]
    assert "username" in config["queue"]
    assert "password" in config["queue"]
    assert "vhost" in config["queue"]
    assert "exchange" in config["queue"]
    assert "queues" in config["queue"] 