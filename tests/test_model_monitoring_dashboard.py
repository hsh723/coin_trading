import pytest
from src.model.model_monitoring_dashboard import ModelMonitoringDashboard
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def monitoring_dashboard():
    config_dir = "./config"
    dashboard_dir = "./dashboard"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "dashboard": {
                "layout": {
                    "rows": 2,
                    "columns": 2,
                    "widgets": [
                        {
                            "type": "metric",
                            "title": "Model Performance",
                            "metrics": ["accuracy", "precision", "recall"]
                        },
                        {
                            "type": "chart",
                            "title": "Prediction Distribution",
                            "chart_type": "histogram"
                        },
                        {
                            "type": "alert",
                            "title": "System Alerts",
                            "threshold": 0.9
                        },
                        {
                            "type": "table",
                            "title": "Recent Predictions",
                            "columns": ["timestamp", "input", "prediction", "confidence"]
                        }
                    ]
                },
                "refresh": {
                    "interval": 60,
                    "timeout": 30
                },
                "export": {
                    "formats": ["csv", "json", "png"],
                    "retention": 7
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_monitoring_dashboard.json"), "w") as f:
        json.dump(config, f)
    
    return ModelMonitoringDashboard(config_dir=config_dir, dashboard_dir=dashboard_dir)

@pytest.fixture
def sample_metrics():
    # 샘플 메트릭 데이터 생성
    return {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.94,
        "f1": 0.935
    }

@pytest.fixture
def sample_predictions():
    # 샘플 예측 데이터 생성
    n_samples = 100
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=n_samples, freq="H"),
        "input": np.random.randn(n_samples, 10).tolist(),
        "prediction": np.random.randint(0, 2, n_samples),
        "confidence": np.random.uniform(0.8, 1.0, n_samples)
    }
    return pd.DataFrame(data)

def test_monitoring_dashboard_initialization(monitoring_dashboard):
    assert monitoring_dashboard is not None
    assert monitoring_dashboard.config_dir == "./config"
    assert monitoring_dashboard.dashboard_dir == "./dashboard"

def test_dashboard_layout(monitoring_dashboard):
    # 대시보드 레이아웃 테스트
    layout = monitoring_dashboard.get_layout()
    
    assert layout is not None
    assert layout.rows == 2
    assert layout.columns == 2
    assert len(layout.widgets) == 4

def test_metric_widget(monitoring_dashboard, sample_metrics):
    # 메트릭 위젯 테스트
    widget = monitoring_dashboard.create_metric_widget("Model Performance", sample_metrics)
    
    assert widget is not None
    assert widget.title == "Model Performance"
    assert "accuracy" in widget.metrics
    assert "precision" in widget.metrics
    assert "recall" in widget.metrics

def test_chart_widget(monitoring_dashboard, sample_predictions):
    # 차트 위젯 테스트
    widget = monitoring_dashboard.create_chart_widget("Prediction Distribution", sample_predictions)
    
    assert widget is not None
    assert widget.title == "Prediction Distribution"
    assert widget.chart_type == "histogram"
    assert widget.data is not None

def test_alert_widget(monitoring_dashboard):
    # 알림 위젯 테스트
    alerts = [
        {"level": "warning", "message": "High latency detected"},
        {"level": "error", "message": "Model accuracy below threshold"}
    ]
    widget = monitoring_dashboard.create_alert_widget("System Alerts", alerts)
    
    assert widget is not None
    assert widget.title == "System Alerts"
    assert len(widget.alerts) == 2
    assert widget.threshold == 0.9

def test_table_widget(monitoring_dashboard, sample_predictions):
    # 테이블 위젯 테스트
    widget = monitoring_dashboard.create_table_widget("Recent Predictions", sample_predictions)
    
    assert widget is not None
    assert widget.title == "Recent Predictions"
    assert "timestamp" in widget.columns
    assert "input" in widget.columns
    assert "prediction" in widget.columns
    assert "confidence" in widget.columns

def test_dashboard_refresh(monitoring_dashboard):
    # 대시보드 새로고침 테스트
    refresh = monitoring_dashboard.setup_refresh()
    
    assert refresh is not None
    assert refresh.interval == 60
    assert refresh.timeout == 30

def test_dashboard_export(monitoring_dashboard):
    # 대시보드 내보내기 테스트
    export = monitoring_dashboard.setup_export()
    
    assert export is not None
    assert "csv" in export.formats
    assert "json" in export.formats
    assert "png" in export.formats
    assert export.retention == 7

def test_dashboard_performance(monitoring_dashboard, sample_metrics, sample_predictions):
    # 대시보드 성능 테스트
    start_time = datetime.now()
    
    # 위젯 생성 및 렌더링
    monitoring_dashboard.create_metric_widget("Model Performance", sample_metrics)
    monitoring_dashboard.create_chart_widget("Prediction Distribution", sample_predictions)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 위젯 생성 및 렌더링을 2초 이내에 완료
    assert processing_time < 2.0

def test_error_handling(monitoring_dashboard):
    # 에러 처리 테스트
    # 잘못된 위젯 타입
    with pytest.raises(ValueError):
        monitoring_dashboard.create_widget("invalid_type", "Test Widget")
    
    # 잘못된 데이터 형식
    with pytest.raises(ValueError):
        monitoring_dashboard.create_chart_widget("Test Chart", None)
    
    # 잘못된 새로고침 설정
    with pytest.raises(ValueError):
        monitoring_dashboard.setup_refresh(interval=-1)

def test_dashboard_configuration(monitoring_dashboard):
    # 대시보드 설정 테스트
    config = monitoring_dashboard.get_configuration()
    
    assert config is not None
    assert "dashboard" in config
    assert "layout" in config["dashboard"]
    assert "refresh" in config["dashboard"]
    assert "export" in config["dashboard"]
    assert "rows" in config["dashboard"]["layout"]
    assert "interval" in config["dashboard"]["refresh"]
    assert "formats" in config["dashboard"]["export"] 