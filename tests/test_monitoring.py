import pytest
from src.monitoring.monitoring_manager import MonitoringManager
import os
import json
import time
import pandas as pd
import numpy as np

@pytest.fixture
def monitoring_manager():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "metrics_interval": 60,
            "alert_thresholds": {
                "cpu_usage": 80,
                "memory_usage": 80,
                "disk_usage": 80,
                "network_latency": 100
            },
            "alert_channels": ["email", "slack"]
        }
    }
    with open(os.path.join(config_dir, "monitoring.json"), "w") as f:
        json.dump(config, f)
    
    return MonitoringManager(config_dir=config_dir, data_dir=data_dir)

def test_monitoring_initialization(monitoring_manager):
    assert monitoring_manager is not None
    assert monitoring_manager.config_dir == "./config"
    assert monitoring_manager.data_dir == "./data"

def test_monitoring_start_stop(monitoring_manager):
    monitoring_manager.start()
    assert monitoring_manager.is_running() is True
    
    monitoring_manager.stop()
    assert monitoring_manager.is_running() is False

def test_system_metrics_collection(monitoring_manager):
    monitoring_manager.start()
    
    # 시스템 메트릭 수집
    metrics = monitoring_manager.collect_system_metrics()
    
    assert metrics is not None
    assert "cpu_usage" in metrics
    assert "memory_usage" in metrics
    assert "disk_usage" in metrics
    assert "network_latency" in metrics
    
    monitoring_manager.stop()

def test_trading_metrics_collection(monitoring_manager):
    monitoring_manager.start()
    
    # 거래 메트릭 수집
    metrics = monitoring_manager.collect_trading_metrics()
    
    assert metrics is not None
    assert "order_count" in metrics
    assert "trade_count" in metrics
    assert "total_volume" in metrics
    assert "total_pnl" in metrics
    
    monitoring_manager.stop()

def test_performance_metrics_collection(monitoring_manager):
    monitoring_manager.start()
    
    # 성능 메트릭 수집
    metrics = monitoring_manager.collect_performance_metrics()
    
    assert metrics is not None
    assert "latency" in metrics
    assert "throughput" in metrics
    assert "error_rate" in metrics
    
    monitoring_manager.stop()

def test_alert_management(monitoring_manager):
    monitoring_manager.start()
    
    # 알림 설정
    monitoring_manager.set_alert_threshold(
        metric="cpu_usage",
        threshold=90
    )
    
    # 알림 확인
    threshold = monitoring_manager.get_alert_threshold("cpu_usage")
    assert threshold == 90
    
    # 알림 채널 추가
    monitoring_manager.add_alert_channel("telegram")
    
    # 알림 채널 확인
    channels = monitoring_manager.get_alert_channels()
    assert "telegram" in channels
    
    monitoring_manager.stop()

def test_metrics_history(monitoring_manager):
    monitoring_manager.start()
    
    # 메트릭 기록 조회
    history = monitoring_manager.get_metrics_history(
        metric="cpu_usage",
        start_time=time.time() - 3600,
        end_time=time.time()
    )
    
    assert history is not None
    assert isinstance(history, pd.DataFrame)
    assert "timestamp" in history.columns
    assert "value" in history.columns
    
    monitoring_manager.stop()

def test_alert_history(monitoring_manager):
    monitoring_manager.start()
    
    # 알림 기록 조회
    history = monitoring_manager.get_alert_history(
        start_time=time.time() - 3600,
        end_time=time.time()
    )
    
    assert history is not None
    assert isinstance(history, pd.DataFrame)
    assert "timestamp" in history.columns
    assert "metric" in history.columns
    assert "value" in history.columns
    assert "threshold" in history.columns
    
    monitoring_manager.stop()

def test_report_generation(monitoring_manager):
    monitoring_manager.start()
    
    # 보고서 생성
    report = monitoring_manager.generate_report(
        start_time=time.time() - 3600,
        end_time=time.time()
    )
    
    assert report is not None
    assert "system_metrics" in report
    assert "trading_metrics" in report
    assert "performance_metrics" in report
    assert "alerts" in report
    
    monitoring_manager.stop()

def test_error_handling(monitoring_manager):
    monitoring_manager.start()
    
    # 잘못된 메트릭 조회 시도
    with pytest.raises(Exception):
        monitoring_manager.get_metrics_history(
            metric="invalid_metric",
            start_time=time.time() - 3600,
            end_time=time.time()
        )
    
    # 에러 통계 확인
    error_stats = monitoring_manager.get_error_stats()
    assert error_stats is not None
    assert error_stats["error_count"] > 0
    
    monitoring_manager.stop() 