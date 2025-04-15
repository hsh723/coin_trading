import pytest
from src.backtest.engine import BacktestEngine
import os
import json
import pandas as pd
import numpy as np

@pytest.fixture
def backtest_engine():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "initial_capital": 10000.0,
            "commission": 0.001,
            "slippage": 0.001
        }
    }
    with open(os.path.join(config_dir, "backtest.json"), "w") as f:
        json.dump(config, f)
    
    return BacktestEngine(config_dir=config_dir, data_dir=data_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="1H")
    data = pd.DataFrame({
        "open": np.random.normal(50000, 1000, len(dates)),
        "high": np.random.normal(51000, 1000, len(dates)),
        "low": np.random.normal(49000, 1000, len(dates)),
        "close": np.random.normal(50500, 1000, len(dates)),
        "volume": np.random.normal(100, 10, len(dates))
    }, index=dates)
    return data

def test_backtest_initialization(backtest_engine):
    assert backtest_engine is not None
    assert backtest_engine.config_dir == "./config"
    assert backtest_engine.data_dir == "./data"

def test_backtest_start_stop(backtest_engine):
    backtest_engine.start()
    assert backtest_engine.is_running() is True
    
    backtest_engine.stop()
    assert backtest_engine.is_running() is False

def test_backtest_data_loading(backtest_engine, sample_data):
    # 데이터 로드
    loaded_data = backtest_engine.load_data(
        symbol="BTCUSDT",
        timeframe="1h",
        start_date="2023-01-01",
        end_date="2023-01-31"
    )
    
    assert loaded_data is not None
    assert len(loaded_data) > 0
    assert "open" in loaded_data.columns
    assert "high" in loaded_data.columns
    assert "low" in loaded_data.columns
    assert "close" in loaded_data.columns
    assert "volume" in loaded_data.columns

def test_backtest_strategy_execution(backtest_engine, sample_data):
    # 전략 실행
    results = backtest_engine.run_strategy(
        data=sample_data,
        strategy={
            "name": "test_strategy",
            "params": {
                "param1": 1.0,
                "param2": 2.0
            }
        }
    )
    
    assert results is not None
    assert "trades" in results
    assert "equity_curve" in results
    assert "performance" in results

def test_backtest_trade_simulation(backtest_engine, sample_data):
    # 거래 시뮬레이션
    trades = backtest_engine.simulate_trades(
        data=sample_data,
        signals=pd.Series(np.random.choice([-1, 0, 1], len(sample_data)), index=sample_data.index)
    )
    
    assert trades is not None
    assert len(trades) > 0
    assert "entry_time" in trades.columns
    assert "exit_time" in trades.columns
    assert "entry_price" in trades.columns
    assert "exit_price" in trades.columns
    assert "pnl" in trades.columns

def test_backtest_performance_analysis(backtest_engine, sample_data):
    # 성과 분석
    performance = backtest_engine.analyze_performance(
        trades=pd.DataFrame({
            "entry_time": sample_data.index[:10],
            "exit_time": sample_data.index[1:11],
            "entry_price": sample_data["close"][:10],
            "exit_price": sample_data["close"][1:11],
            "pnl": np.random.normal(0, 100, 10)
        })
    )
    
    assert performance is not None
    assert "total_return" in performance
    assert "sharpe_ratio" in performance
    assert "max_drawdown" in performance
    assert "win_rate" in performance

def test_backtest_risk_analysis(backtest_engine, sample_data):
    # 리스크 분석
    risk_metrics = backtest_engine.analyze_risk(
        trades=pd.DataFrame({
            "entry_time": sample_data.index[:10],
            "exit_time": sample_data.index[1:11],
            "entry_price": sample_data["close"][:10],
            "exit_price": sample_data["close"][1:11],
            "pnl": np.random.normal(0, 100, 10)
        })
    )
    
    assert risk_metrics is not None
    assert "var" in risk_metrics
    assert "cvar" in risk_metrics
    assert "volatility" in risk_metrics
    assert "beta" in risk_metrics

def test_backtest_optimization(backtest_engine, sample_data):
    # 최적화
    optimized_params = backtest_engine.optimize_strategy(
        data=sample_data,
        strategy={
            "name": "test_strategy",
            "param_ranges": {
                "param1": (0.5, 2.0),
                "param2": (1.0, 3.0)
            }
        }
    )
    
    assert optimized_params is not None
    assert "param1" in optimized_params
    assert "param2" in optimized_params
    assert "performance" in optimized_params

def test_backtest_report_generation(backtest_engine, sample_data):
    # 보고서 생성
    report = backtest_engine.generate_report(
        trades=pd.DataFrame({
            "entry_time": sample_data.index[:10],
            "exit_time": sample_data.index[1:11],
            "entry_price": sample_data["close"][:10],
            "exit_price": sample_data["close"][1:11],
            "pnl": np.random.normal(0, 100, 10)
        })
    )
    
    assert report is not None
    assert "summary" in report
    assert "performance" in report
    assert "risk_metrics" in report
    assert "trades" in report

def test_backtest_error_handling(backtest_engine):
    # 잘못된 데이터로 테스트 시도
    with pytest.raises(Exception):
        backtest_engine.run_strategy(
            data=None,
            strategy={
                "name": "test_strategy",
                "params": {}
            }
        )
    
    # 에러 통계 확인
    error_stats = backtest_engine.get_error_stats()
    assert error_stats is not None
    assert error_stats["error_count"] > 0 