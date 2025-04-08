import pytest
import pandas as pd
from src.risk.risk_manager import RiskManager

def test_risk_manager_initialization():
    """리스크 관리자 초기화 테스트"""
    risk_manager = RiskManager()
    assert risk_manager is not None
    assert risk_manager.max_position_size == 0.1
    assert risk_manager.max_drawdown == 0.2
    assert risk_manager.stop_loss == 0.05
    assert risk_manager.take_profit == 0.1
    assert risk_manager.trailing_stop == 0.03

def test_position_sizing(sample_market_data):
    """포지션 사이징 테스트"""
    risk_manager = RiskManager()
    
    # 기본 포지션 사이즈 계산
    position_size = risk_manager.calculate_position_size(
        balance=10000,
        price=100,
        risk_per_trade=0.02
    )
    assert position_size > 0
    assert position_size <= risk_manager.max_position_size * 10000 / 100
    
    # 리스크 제한 테스트
    large_position = risk_manager.calculate_position_size(
        balance=10000,
        price=100,
        risk_per_trade=0.5  # 매우 높은 리스크
    )
    assert large_position <= risk_manager.max_position_size * 10000 / 100

def test_stop_loss_calculation(sample_market_data):
    """손절가 계산 테스트"""
    risk_manager = RiskManager()
    
    # 매수 포지션 손절가
    stop_loss = risk_manager.calculate_stop_loss(
        entry_price=100,
        side='buy',
        stop_loss_pct=0.05
    )
    assert stop_loss == 95
    
    # 매도 포지션 손절가
    stop_loss = risk_manager.calculate_stop_loss(
        entry_price=100,
        side='sell',
        stop_loss_pct=0.05
    )
    assert stop_loss == 105

def test_take_profit_calculation(sample_market_data):
    """익절가 계산 테스트"""
    risk_manager = RiskManager()
    
    # 매수 포지션 익절가
    take_profit = risk_manager.calculate_take_profit(
        entry_price=100,
        side='buy',
        take_profit_pct=0.1
    )
    assert take_profit == 110
    
    # 매도 포지션 익절가
    take_profit = risk_manager.calculate_take_profit(
        entry_price=100,
        side='sell',
        take_profit_pct=0.1
    )
    assert take_profit == 90

def test_trailing_stop_update(sample_market_data):
    """트레일링 스탑 업데이트 테스트"""
    risk_manager = RiskManager()
    
    # 매수 포지션 트레일링 스탑
    current_price = 110
    trailing_stop = risk_manager.update_trailing_stop(
        current_price=current_price,
        highest_price=120,
        lowest_price=100,
        side='buy'
    )
    assert trailing_stop == 116.4  # 120 * (1 - 0.03)
    
    # 매도 포지션 트레일링 스탑
    trailing_stop = risk_manager.update_trailing_stop(
        current_price=90,
        highest_price=100,
        lowest_price=80,
        side='sell'
    )
    assert trailing_stop == 82.8  # 80 * (1 + 0.03)

def test_drawdown_monitoring(sample_market_data):
    """낙폭 모니터링 테스트"""
    risk_manager = RiskManager()
    
    # 초기 자본금 설정
    initial_balance = 10000
    current_balance = 8000
    
    # 낙폭 계산
    drawdown = risk_manager.calculate_drawdown(initial_balance, current_balance)
    assert drawdown == 0.2
    
    # 낙폭 제한 초과 테스트
    assert risk_manager.check_drawdown_limit(drawdown) == False

def test_risk_limits(sample_market_data):
    """리스크 제한 테스트"""
    risk_manager = RiskManager()
    
    # 포지션 사이즈 제한
    assert risk_manager.check_position_size_limit(0.05) == True
    assert risk_manager.check_position_size_limit(0.2) == False
    
    # 낙폭 제한
    assert risk_manager.check_drawdown_limit(0.1) == True
    assert risk_manager.check_drawdown_limit(0.3) == False
    
    # 리스크 대비 보상 비율
    assert risk_manager.check_risk_reward_ratio(0.05, 0.1) == True
    assert risk_manager.check_risk_reward_ratio(0.1, 0.05) == False

def test_risk_metrics(sample_market_data):
    """리스크 지표 계산 테스트"""
    risk_manager = RiskManager()
    
    # 샤프 비율 계산
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    sharpe_ratio = risk_manager.calculate_sharpe_ratio(returns)
    assert isinstance(sharpe_ratio, float)
    
    # 최대 낙폭 계산
    drawdown = risk_manager.calculate_max_drawdown(returns)
    assert isinstance(drawdown, float)
    assert 0 <= drawdown <= 1
    
    # 변동성 계산
    volatility = risk_manager.calculate_volatility(returns)
    assert isinstance(volatility, float)
    assert volatility >= 0

def test_risk_adjustment(sample_market_data):
    """리스크 조정 테스트"""
    risk_manager = RiskManager()
    
    # 시장 변동성에 따른 리스크 조정
    volatility = 0.2
    adjusted_risk = risk_manager.adjust_risk_for_volatility(
        base_risk=0.02,
        volatility=volatility
    )
    assert adjusted_risk <= 0.02
    
    # 계좌 크기에 따른 리스크 조정
    account_size = 5000
    adjusted_position = risk_manager.adjust_position_for_account_size(
        base_position=0.1,
        account_size=account_size
    )
    assert adjusted_position <= 0.1 