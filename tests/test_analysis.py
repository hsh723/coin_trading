import pytest
from src.analysis.analysis_manager import AnalysisManager
import os
import json
import pandas as pd
import numpy as np

@pytest.fixture
def analysis_manager():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "analysis_methods": ["technical", "fundamental", "sentiment"],
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "indicators": ["sma", "ema", "rsi", "macd", "bollinger_bands"]
        }
    }
    with open(os.path.join(config_dir, "analysis.json"), "w") as f:
        json.dump(config, f)
    
    return AnalysisManager(config_dir=config_dir, data_dir=data_dir)

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

def test_analysis_initialization(analysis_manager):
    assert analysis_manager is not None
    assert analysis_manager.config_dir == "./config"
    assert analysis_manager.data_dir == "./data"

def test_technical_analysis(analysis_manager, sample_data):
    # 기술적 분석
    indicators = analysis_manager.calculate_technical_indicators(
        data=sample_data,
        indicators=["sma", "ema", "rsi", "macd"]
    )
    
    assert indicators is not None
    assert "sma" in indicators
    assert "ema" in indicators
    assert "rsi" in indicators
    assert "macd" in indicators

def test_fundamental_analysis(analysis_manager):
    # 기본적 분석
    metrics = analysis_manager.calculate_fundamental_metrics(
        symbol="BTCUSDT",
        timeframe="1d"
    )
    
    assert metrics is not None
    assert "market_cap" in metrics
    assert "volume_24h" in metrics
    assert "circulating_supply" in metrics
    assert "total_supply" in metrics

def test_sentiment_analysis(analysis_manager):
    # 감성 분석
    sentiment = analysis_manager.analyze_sentiment(
        symbol="BTCUSDT",
        timeframe="1d"
    )
    
    assert sentiment is not None
    assert "score" in sentiment
    assert "magnitude" in sentiment
    assert "sentiment" in sentiment

def test_correlation_analysis(analysis_manager, sample_data):
    # 상관관계 분석
    correlations = analysis_manager.calculate_correlations(
        data=sample_data,
        pairs=[("BTCUSDT", "ETHUSDT"), ("BTCUSDT", "BNBUSDT")]
    )
    
    assert correlations is not None
    assert isinstance(correlations, pd.DataFrame)
    assert "BTCUSDT-ETHUSDT" in correlations.columns
    assert "BTCUSDT-BNBUSDT" in correlations.columns

def test_volatility_analysis(analysis_manager, sample_data):
    # 변동성 분석
    volatility = analysis_manager.calculate_volatility(
        data=sample_data,
        window=20
    )
    
    assert volatility is not None
    assert isinstance(volatility, pd.Series)
    assert len(volatility) == len(sample_data)

def test_market_regime_analysis(analysis_manager, sample_data):
    # 시장 레짐 분석
    regime = analysis_manager.identify_market_regime(
        data=sample_data,
        window=20
    )
    
    assert regime is not None
    assert isinstance(regime, pd.Series)
    assert len(regime) == len(sample_data)
    assert all(regime.isin(["trending", "ranging", "volatile"]))

def test_pattern_recognition(analysis_manager, sample_data):
    # 패턴 인식
    patterns = analysis_manager.identify_patterns(
        data=sample_data,
        patterns=["head_and_shoulders", "double_top", "double_bottom"]
    )
    
    assert patterns is not None
    assert isinstance(patterns, pd.DataFrame)
    assert "pattern" in patterns.columns
    assert "start_time" in patterns.columns
    assert "end_time" in patterns.columns

def test_risk_metrics_calculation(analysis_manager, sample_data):
    # 리스크 메트릭 계산
    risk_metrics = analysis_manager.calculate_risk_metrics(
        data=sample_data,
        window=20
    )
    
    assert risk_metrics is not None
    assert "var" in risk_metrics
    assert "cvar" in risk_metrics
    assert "sharpe_ratio" in risk_metrics
    assert "sortino_ratio" in risk_metrics

def test_report_generation(analysis_manager, sample_data):
    # 분석 보고서 생성
    report = analysis_manager.generate_analysis_report(
        data=sample_data,
        symbol="BTCUSDT",
        timeframe="1h"
    )
    
    assert report is not None
    assert "technical_analysis" in report
    assert "fundamental_analysis" in report
    assert "sentiment_analysis" in report
    assert "risk_metrics" in report
    assert "recommendations" in report

def test_error_handling(analysis_manager):
    # 잘못된 데이터로 분석 시도
    with pytest.raises(Exception):
        analysis_manager.calculate_technical_indicators(
            data=None,
            indicators=["sma"]
        )
    
    # 에러 통계 확인
    error_stats = analysis_manager.get_error_stats()
    assert error_stats is not None
    assert error_stats["error_count"] > 0 