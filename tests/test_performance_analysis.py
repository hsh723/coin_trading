import pytest
from src.performance.performance_analyzer import PerformanceAnalyzer
import os
import json
import pandas as pd
import numpy as np

@pytest.fixture
def performance_analyzer():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "metrics": {
                "returns": ["total_return", "daily_return", "monthly_return"],
                "risk": ["sharpe_ratio", "sortino_ratio", "max_drawdown"],
                "trade": ["win_rate", "profit_factor", "average_trade"]
            },
            "benchmarks": ["BTC", "ETH", "BNB"],
            "timeframes": ["1d", "1w", "1m", "3m", "6m", "1y"]
        }
    }
    with open(os.path.join(config_dir, "performance_analysis.json"), "w") as f:
        json.dump(config, f)
    
    return PerformanceAnalyzer(config_dir=config_dir, data_dir=data_dir)

@pytest.fixture
def sample_trades():
    # 샘플 거래 데이터 생성
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="1D")
    trades = pd.DataFrame({
        "entry_time": dates,
        "exit_time": dates + pd.Timedelta(hours=1),
        "symbol": ["BTCUSDT"] * len(dates),
        "side": np.random.choice(["buy", "sell"], len(dates)),
        "entry_price": np.random.normal(50000, 1000, len(dates)),
        "exit_price": np.random.normal(50500, 1000, len(dates)),
        "quantity": np.random.normal(0.1, 0.01, len(dates)),
        "pnl": np.random.normal(50, 10, len(dates))
    })
    return trades

def test_performance_analyzer_initialization(performance_analyzer):
    assert performance_analyzer is not None
    assert performance_analyzer.config_dir == "./config"
    assert performance_analyzer.data_dir == "./data"

def test_performance_analyzer_start_stop(performance_analyzer):
    performance_analyzer.start()
    assert performance_analyzer.is_running() is True
    
    performance_analyzer.stop()
    assert performance_analyzer.is_running() is False

def test_returns_analysis(performance_analyzer, sample_trades):
    performance_analyzer.start()
    
    # 수익률 분석
    returns = performance_analyzer.analyze_returns(sample_trades)
    
    assert returns is not None
    assert "total_return" in returns
    assert "daily_returns" in returns
    assert "monthly_returns" in returns
    assert "annualized_return" in returns
    
    performance_analyzer.stop()

def test_risk_analysis(performance_analyzer, sample_trades):
    performance_analyzer.start()
    
    # 리스크 분석
    risk_metrics = performance_analyzer.analyze_risk(sample_trades)
    
    assert risk_metrics is not None
    assert "sharpe_ratio" in risk_metrics
    assert "sortino_ratio" in risk_metrics
    assert "max_drawdown" in risk_metrics
    assert "volatility" in risk_metrics
    
    performance_analyzer.stop()

def test_trade_analysis(performance_analyzer, sample_trades):
    performance_analyzer.start()
    
    # 거래 분석
    trade_metrics = performance_analyzer.analyze_trades(sample_trades)
    
    assert trade_metrics is not None
    assert "win_rate" in trade_metrics
    assert "profit_factor" in trade_metrics
    assert "average_trade" in trade_metrics
    assert "average_win" in trade_metrics
    assert "average_loss" in trade_metrics
    
    performance_analyzer.stop()

def test_benchmark_comparison(performance_analyzer, sample_trades):
    performance_analyzer.start()
    
    # 벤치마크 비교
    comparison = performance_analyzer.compare_with_benchmarks(
        trades=sample_trades,
        benchmarks=["BTC", "ETH"]
    )
    
    assert comparison is not None
    assert "BTC" in comparison
    assert "ETH" in comparison
    assert "relative_performance" in comparison
    
    performance_analyzer.stop()

def test_performance_attribution(performance_analyzer, sample_trades):
    performance_analyzer.start()
    
    # 성과 귀속 분석
    attribution = performance_analyzer.analyze_performance_attribution(sample_trades)
    
    assert attribution is not None
    assert "market_contribution" in attribution
    assert "strategy_contribution" in attribution
    assert "timing_contribution" in attribution
    
    performance_analyzer.stop()

def test_performance_report_generation(performance_analyzer, sample_trades):
    performance_analyzer.start()
    
    # 성과 보고서 생성
    report = performance_analyzer.generate_performance_report(
        trades=sample_trades,
        start_date="2023-01-01",
        end_date="2023-01-31"
    )
    
    assert report is not None
    assert "summary" in report
    assert "returns_analysis" in report
    assert "risk_analysis" in report
    assert "trade_analysis" in report
    assert "benchmark_comparison" in report
    assert "recommendations" in report
    
    performance_analyzer.stop()

def test_error_handling(performance_analyzer):
    performance_analyzer.start()
    
    # 잘못된 데이터로 분석 시도
    with pytest.raises(Exception):
        performance_analyzer.analyze_returns(None)
    
    # 에러 통계 확인
    error_stats = performance_analyzer.get_error_stats()
    assert error_stats is not None
    assert error_stats["error_count"] > 0
    
    performance_analyzer.stop() 