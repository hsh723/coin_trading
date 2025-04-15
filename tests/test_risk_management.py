import pytest
from src.risk_management.risk_manager import RiskManager
import os
import json
import pandas as pd
import numpy as np

@pytest.fixture
def risk_manager():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "position_limits": {
                "max_position_size": 1.0,
                "max_leverage": 20,
                "max_daily_trades": 100
            },
            "risk_limits": {
                "max_drawdown": 0.1,
                "max_loss_per_trade": 0.02,
                "max_daily_loss": 0.05,
                "var_limit": 0.03,
                "cvar_limit": 0.04
            },
            "monitoring": {
                "check_interval": 60,
                "alert_threshold": 0.8
            }
        }
    }
    with open(os.path.join(config_dir, "risk_management.json"), "w") as f:
        json.dump(config, f)
    
    return RiskManager(config_dir=config_dir, data_dir=data_dir)

def test_risk_manager_initialization(risk_manager):
    assert risk_manager is not None
    assert risk_manager.config_dir == "./config"
    assert risk_manager.data_dir == "./data"

def test_risk_manager_start_stop(risk_manager):
    risk_manager.start()
    assert risk_manager.is_running() is True
    
    risk_manager.stop()
    assert risk_manager.is_running() is False

def test_position_risk_check(risk_manager):
    risk_manager.start()
    
    # 포지션 리스크 체크
    risk_check = risk_manager.check_position_risk(
        symbol="BTCUSDT",
        position_size=0.5,
        leverage=10,
        entry_price=50000.0,
        current_price=51000.0
    )
    
    assert risk_check is not None
    assert "is_allowed" in risk_check
    assert "reason" in risk_check
    assert "risk_metrics" in risk_check
    
    risk_manager.stop()

def test_trade_risk_check(risk_manager):
    risk_manager.start()
    
    # 거래 리스크 체크
    risk_check = risk_manager.check_trade_risk(
        symbol="BTCUSDT",
        order_type="limit",
        side="buy",
        quantity=0.1,
        price=50000.0
    )
    
    assert risk_check is not None
    assert "is_allowed" in risk_check
    assert "reason" in risk_check
    assert "risk_metrics" in risk_check
    
    risk_manager.stop()

def test_portfolio_risk_check(risk_manager):
    risk_manager.start()
    
    # 포트폴리오 리스크 체크
    portfolio = {
        "BTCUSDT": {
            "position_size": 0.5,
            "entry_price": 50000.0,
            "current_price": 51000.0
        },
        "ETHUSDT": {
            "position_size": 1.0,
            "entry_price": 3000.0,
            "current_price": 3100.0
        }
    }
    
    risk_check = risk_manager.check_portfolio_risk(portfolio)
    
    assert risk_check is not None
    assert "is_allowed" in risk_check
    assert "reason" in risk_check
    assert "risk_metrics" in risk_check
    
    risk_manager.stop()

def test_risk_metrics_calculation(risk_manager):
    risk_manager.start()
    
    # 리스크 메트릭 계산
    metrics = risk_manager.calculate_risk_metrics(
        symbol="BTCUSDT",
        position_size=0.5,
        entry_price=50000.0,
        current_price=51000.0
    )
    
    assert metrics is not None
    assert "var" in metrics
    assert "cvar" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    
    risk_manager.stop()

def test_risk_limits_management(risk_manager):
    risk_manager.start()
    
    # 리스크 한도 설정
    result = risk_manager.set_risk_limits(
        symbol="BTCUSDT",
        max_position_size=1.0,
        max_leverage=20,
        max_daily_trades=100,
        max_drawdown=0.1,
        max_loss_per_trade=0.02
    )
    
    assert result is True
    
    # 리스크 한도 확인
    limits = risk_manager.get_risk_limits("BTCUSDT")
    assert limits is not None
    assert limits["max_position_size"] == 1.0
    assert limits["max_leverage"] == 20
    assert limits["max_daily_trades"] == 100
    
    risk_manager.stop()

def test_risk_monitoring(risk_manager):
    risk_manager.start()
    
    # 리스크 모니터링 시작
    risk_manager.start_monitoring()
    
    # 모니터링 상태 확인
    status = risk_manager.get_monitoring_status()
    assert status is not None
    assert "is_running" in status
    assert "last_check" in status
    assert "alerts" in status
    
    risk_manager.stop()

def test_risk_report_generation(risk_manager):
    risk_manager.start()
    
    # 리스크 보고서 생성
    report = risk_manager.generate_risk_report(
        symbol="BTCUSDT",
        start_time="2023-01-01",
        end_time="2023-01-31"
    )
    
    assert report is not None
    assert "position_risk" in report
    assert "trade_risk" in report
    assert "portfolio_risk" in report
    assert "risk_metrics" in report
    assert "recommendations" in report
    
    risk_manager.stop()

def test_error_handling(risk_manager):
    risk_manager.start()
    
    # 잘못된 리스크 체크 시도
    with pytest.raises(Exception):
        risk_manager.check_position_risk(
            symbol="INVALID",
            position_size=1.0,
            leverage=10,
            entry_price=50000.0,
            current_price=51000.0
        )
    
    # 에러 통계 확인
    error_stats = risk_manager.get_error_stats()
    assert error_stats is not None
    assert error_stats["error_count"] > 0
    
    risk_manager.stop() 