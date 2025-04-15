import pytest
from src.model.model_monitor import ModelMonitor
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def model_monitor():
    config_dir = "./config"
    monitor_dir = "./monitor"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "monitoring": {
                "metrics": {
                    "performance": ["accuracy", "precision", "recall", "f1"],
                    "drift": ["psi", "ks", "chi_square"],
                    "data_quality": ["missing_rate", "outlier_rate", "duplicate_rate"]
                },
                "thresholds": {
                    "accuracy": 0.8,
                    "psi": 0.1,
                    "missing_rate": 0.05
                },
                "alerts": {
                    "email": "admin@example.com",
                    "slack": "monitoring-channel",
                    "threshold": 0.9
                },
                "logging": {
                    "interval": 60,
                    "retention": 30
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_monitor.json"), "w") as f:
        json.dump(config, f)
    
    return ModelMonitor(config_dir=config_dir, monitor_dir=monitor_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    n_samples = 1000
    n_features = 10
    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "target": np.random.randint(0, 2, n_samples)
    }
    return pd.DataFrame(data)

@pytest.fixture
def reference_data():
    # 참조 데이터 생성
    n_samples = 1000
    n_features = 10
    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "target": np.random.randint(0, 2, n_samples)
    }
    return pd.DataFrame(data)

def test_model_monitor_initialization(model_monitor):
    assert model_monitor is not None
    assert model_monitor.config_dir == "./config"
    assert model_monitor.monitor_dir == "./monitor"

def test_performance_metrics(model_monitor, sample_data):
    # 성능 메트릭 테스트
    metrics = model_monitor.calculate_performance_metrics(sample_data)
    
    assert metrics is not None
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert all(0 <= v <= 1 for v in metrics.values())

def test_drift_metrics(model_monitor, sample_data, reference_data):
    # 데이터 드리프트 메트릭 테스트
    drift_metrics = model_monitor.calculate_drift_metrics(sample_data, reference_data)
    
    assert drift_metrics is not None
    assert "psi" in drift_metrics
    assert "ks" in drift_metrics
    assert "chi_square" in drift_metrics
    assert all(v >= 0 for v in drift_metrics.values())

def test_data_quality_metrics(model_monitor, sample_data):
    # 데이터 품질 메트릭 테스트
    quality_metrics = model_monitor.calculate_data_quality_metrics(sample_data)
    
    assert quality_metrics is not None
    assert "missing_rate" in quality_metrics
    assert "outlier_rate" in quality_metrics
    assert "duplicate_rate" in quality_metrics
    assert all(0 <= v <= 1 for v in quality_metrics.values())

def test_threshold_alerting(model_monitor, sample_data):
    # 임계값 알림 테스트
    alerts = model_monitor.check_thresholds(sample_data)
    
    assert alerts is not None
    assert isinstance(alerts, list)
    assert all(isinstance(alert, dict) for alert in alerts)
    assert all("metric" in alert and "value" in alert and "threshold" in alert for alert in alerts)

def test_alert_notification(model_monitor):
    # 알림 전송 테스트
    alert = {
        "metric": "accuracy",
        "value": 0.75,
        "threshold": 0.8
    }
    
    # 이메일 알림
    email_sent = model_monitor.send_email_alert(alert)
    assert email_sent is True
    
    # 슬랙 알림
    slack_sent = model_monitor.send_slack_alert(alert)
    assert slack_sent is True

def test_metric_logging(model_monitor, sample_data):
    # 메트릭 로깅 테스트
    log_file = model_monitor.log_metrics(sample_data)
    
    assert log_file is not None
    assert os.path.exists(log_file)
    assert os.path.getsize(log_file) > 0

def test_metric_visualization(model_monitor, sample_data):
    # 메트릭 시각화 테스트
    plot_file = model_monitor.visualize_metrics(sample_data)
    
    assert plot_file is not None
    assert os.path.exists(plot_file)
    assert os.path.getsize(plot_file) > 0

def test_monitoring_performance(model_monitor, sample_data):
    # 모니터링 성능 테스트
    start_time = datetime.now()
    
    # 메트릭 계산 및 로깅
    model_monitor.calculate_performance_metrics(sample_data)
    model_monitor.log_metrics(sample_data)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 1000개 샘플에 대한 모니터링을 5초 이내에 완료
    assert processing_time < 5.0

def test_error_handling(model_monitor):
    # 에러 처리 테스트
    # 잘못된 데이터
    with pytest.raises(ValueError):
        model_monitor.calculate_performance_metrics(None)
    
    # 잘못된 임계값
    with pytest.raises(ValueError):
        model_monitor.check_thresholds(pd.DataFrame(), thresholds={"invalid": 0.5})
    
    # 잘못된 알림 설정
    with pytest.raises(ValueError):
        model_monitor.send_email_alert(None)

def test_monitoring_configuration(model_monitor):
    # 모니터링 설정 테스트
    config = model_monitor.get_configuration()
    
    assert config is not None
    assert "monitoring" in config
    assert "metrics" in config["monitoring"]
    assert "thresholds" in config["monitoring"]
    assert "alerts" in config["monitoring"]
    assert "logging" in config["monitoring"]
    assert "performance" in config["monitoring"]["metrics"]
    assert "drift" in config["monitoring"]["metrics"]
    assert "data_quality" in config["monitoring"]["metrics"] 