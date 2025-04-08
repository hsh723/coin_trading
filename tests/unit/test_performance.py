"""
성과 분석기 테스트
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from src.analysis.performance import PerformanceAnalyzer

@pytest.fixture
def analyzer():
    """테스트용 성과 분석기 생성"""
    return PerformanceAnalyzer(data_storage_path="test_data")

@pytest.fixture
def sample_trades():
    """샘플 거래 데이터 생성"""
    trades = []
    base_time = datetime.now()
    
    # 수익 거래
    for i in range(5):
        trades.append({
            'timestamp': (base_time + timedelta(days=i)).isoformat(),
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.1,
            'price': 50000.0,
            'fee': 2.0,
            'profit': 100.0,
            'balance': 10000.0 + i * 100.0
        })
    
    # 손실 거래
    for i in range(3):
        trades.append({
            'timestamp': (base_time + timedelta(days=i+5)).isoformat(),
            'symbol': 'ETH/USDT',
            'side': 'sell',
            'amount': 1.0,
            'price': 3000.0,
            'fee': 1.2,
            'profit': -50.0,
            'balance': 10000.0 + 500.0 - i * 50.0
        })
    
    return trades

@pytest.fixture
def save_sample_trades(analyzer, sample_trades):
    """샘플 거래 데이터 저장"""
    file_path = os.path.join(analyzer.data_storage_path, 'trades.json')
    with open(file_path, 'w') as f:
        json.dump(sample_trades, f)

def test_initialization(analyzer):
    """초기화 테스트"""
    assert analyzer.data_storage_path == "test_data"
    assert analyzer.trades_df is None
    assert os.path.exists(analyzer.data_storage_path)

def test_load_trade_history(analyzer, save_sample_trades):
    """거래 내역 로드 테스트"""
    success = analyzer.load_trade_history()
    assert success is True
    assert analyzer.trades_df is not None
    assert len(analyzer.trades_df) == 8

def test_calculate_returns(analyzer, save_sample_trades):
    """수익률 계산 테스트"""
    analyzer.load_trade_history()
    
    # 일별 수익률
    daily_returns = analyzer.calculate_returns('daily')
    assert len(daily_returns) > 0
    assert daily_returns.sum() == 350.0  # 5 * 100 - 3 * 50
    
    # 주별 수익률
    weekly_returns = analyzer.calculate_returns('weekly')
    assert len(weekly_returns) > 0
    
    # 월별 수익률
    monthly_returns = analyzer.calculate_returns('monthly')
    assert len(monthly_returns) > 0

def test_calculate_win_rate(analyzer, save_sample_trades):
    """승률 계산 테스트"""
    analyzer.load_trade_history()
    win_rate_info = analyzer.calculate_win_rate()
    
    assert win_rate_info['total_trades'] == 8
    assert win_rate_info['winning_trades'] == 5
    assert win_rate_info['losing_trades'] == 3
    assert win_rate_info['win_rate'] == 5/8
    assert win_rate_info['profit_factor'] > 1

def test_calculate_mdd(analyzer, save_sample_trades):
    """최대 낙폭 계산 테스트"""
    analyzer.load_trade_history()
    mdd = analyzer.calculate_mdd()
    assert mdd < 0  # 손실이 있으므로 MDD는 음수

def test_calculate_sharpe_ratio(analyzer, save_sample_trades):
    """샤프 비율 계산 테스트"""
    analyzer.load_trade_history()
    sharpe_ratio = analyzer.calculate_sharpe_ratio()
    assert isinstance(sharpe_ratio, float)

def test_calculate_calmar_ratio(analyzer, save_sample_trades):
    """칼마 비율 계산 테스트"""
    analyzer.load_trade_history()
    calmar_ratio = analyzer.calculate_calmar_ratio()
    assert isinstance(calmar_ratio, float)

def test_generate_report_json(analyzer, save_sample_trades):
    """JSON 형식 보고서 생성 테스트"""
    analyzer.load_trade_history()
    success = analyzer.generate_report(format='json')
    
    assert success is True
    report_path = os.path.join(analyzer.data_storage_path, 'performance_report.json')
    assert os.path.exists(report_path)
    
    with open(report_path, 'r') as f:
        report = json.load(f)
        assert 'summary' in report
        assert 'returns' in report
        assert 'performance_metrics' in report

def test_generate_report_csv(analyzer, save_sample_trades):
    """CSV 형식 보고서 생성 테스트"""
    analyzer.load_trade_history()
    success = analyzer.generate_report(format='csv')
    
    assert success is True
    returns_path = os.path.join(analyzer.data_storage_path, 'daily_returns.csv')
    metrics_path = os.path.join(analyzer.data_storage_path, 'performance_metrics.csv')
    
    assert os.path.exists(returns_path)
    assert os.path.exists(metrics_path)
    
    returns_df = pd.read_csv(returns_path)
    metrics_df = pd.read_csv(metrics_path)
    
    assert len(returns_df) > 0
    assert len(metrics_df) == 3  # MDD, Sharpe Ratio, Calmar Ratio 