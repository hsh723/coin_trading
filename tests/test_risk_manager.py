"""
리스크 관리 모듈 테스트
"""

import pytest
import pandas as pd
import numpy as np
from src.risk.manager import RiskManager

@pytest.fixture
def risk_manager():
    """리스크 관리자 인스턴스 생성"""
    return RiskManager(
        initial_capital=10000,
        max_position_size=0.1,
        max_drawdown=0.2,
        daily_loss_limit=0.05,
        max_leverage=10
    )

def test_initialization(risk_manager):
    """초기화 테스트"""
    assert risk_manager.initial_capital == 10000
    assert risk_manager.max_position_size == 0.1
    assert risk_manager.max_drawdown == 0.2
    assert risk_manager.daily_loss_limit == 0.05
    assert risk_manager.max_leverage == 10
    assert risk_manager.current_capital == 10000
    assert risk_manager.positions == {}
    assert risk_manager.daily_pnl == 0
    assert risk_manager.max_capital == 10000
    assert risk_manager.min_capital == 10000

def test_position_size_calculation(risk_manager):
    """포지션 크기 계산 테스트"""
    # 정상적인 경우
    position_size = risk_manager.calculate_position_size(
        price=100,
        stop_loss=0.02,
        risk_per_trade=0.01
    )
    assert position_size > 0
    assert position_size <= risk_manager.current_capital * risk_manager.max_position_size
    
    # 최대 포지션 크기 제한
    position_size = risk_manager.calculate_position_size(
        price=100,
        stop_loss=0.01,
        risk_per_trade=0.1
    )
    assert position_size <= risk_manager.current_capital * risk_manager.max_position_size

def test_stop_loss_calculation(risk_manager):
    """손절가 계산 테스트"""
    # 롱 포지션
    stop_loss = risk_manager.calculate_stop_loss(
        entry_price=100,
        position_type='long',
        stop_loss_pct=0.02
    )
    assert stop_loss == 98
    
    # 숏 포지션
    stop_loss = risk_manager.calculate_stop_loss(
        entry_price=100,
        position_type='short',
        stop_loss_pct=0.02
    )
    assert stop_loss == 102

def test_take_profit_calculation(risk_manager):
    """익절가 계산 테스트"""
    # 롱 포지션
    take_profit = risk_manager.calculate_take_profit(
        entry_price=100,
        position_type='long',
        take_profit_pct=0.04
    )
    assert take_profit == 104
    
    # 숏 포지션
    take_profit = risk_manager.calculate_take_profit(
        entry_price=100,
        position_type='short',
        take_profit_pct=0.04
    )
    assert take_profit == 96

def test_position_management(risk_manager):
    """포지션 관리 테스트"""
    # 포지션 진입
    risk_manager.open_position(
        symbol='BTC/USDT',
        position_type='long',
        size=0.1,
        entry_price=100,
        stop_loss=98,
        take_profit=104
    )
    assert 'BTC/USDT' in risk_manager.positions
    assert risk_manager.positions['BTC/USDT']['size'] == 0.1
    
    # 포지션 업데이트
    risk_manager.update_position(
        symbol='BTC/USDT',
        current_price=102,
        unrealized_pnl=20
    )
    assert risk_manager.positions['BTC/USDT']['unrealized_pnl'] == 20
    
    # 포지션 청산
    risk_manager.close_position('BTC/USDT', 104)
    assert 'BTC/USDT' not in risk_manager.positions

def test_risk_limits(risk_manager):
    """리스크 한도 테스트"""
    # 일일 손실 한도
    risk_manager.update_daily_pnl(-600)  # 6% 손실
    assert not risk_manager.check_risk_limits()
    
    # 최대 손실폭 한도
    risk_manager.current_capital = 8000  # 20% 손실
    assert not risk_manager.check_risk_limits()
    
    # 레버리지 한도
    position_size = risk_manager.calculate_position_size(
        price=100,
        stop_loss=0.01,
        risk_per_trade=0.1
    )
    leverage = position_size * 100 / risk_manager.current_capital
    assert leverage <= risk_manager.max_leverage

def test_pnl_calculation(risk_manager):
    """손익 계산 테스트"""
    # 포지션 진입
    risk_manager.open_position(
        symbol='BTC/USDT',
        position_type='long',
        size=0.1,
        entry_price=100,
        stop_loss=98,
        take_profit=104
    )
    
    # 수익 발생
    risk_manager.update_position('BTC/USDT', 102, 20)
    assert risk_manager.calculate_total_pnl() == 20
    
    # 손실 발생
    risk_manager.update_position('BTC/USDT', 99, -10)
    assert risk_manager.calculate_total_pnl() == -10

def test_capital_management(risk_manager):
    """자본금 관리 테스트"""
    # 수익 발생
    risk_manager.update_capital(10500)
    assert risk_manager.max_capital == 10500
    
    # 손실 발생
    risk_manager.update_capital(9500)
    assert risk_manager.min_capital == 9500
    
    # 현재 자본금 업데이트
    assert risk_manager.current_capital == 9500

def test_empty_positions(risk_manager):
    """빈 포지션 처리 테스트"""
    assert risk_manager.calculate_total_pnl() == 0
    assert risk_manager.get_position('BTC/USDT') is None
    risk_manager.close_position('BTC/USDT', 100)  # 에러 발생하지 않아야 함 