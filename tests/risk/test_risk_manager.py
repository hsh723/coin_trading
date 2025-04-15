"""
리스크 관리 시스템 테스트 모듈
"""

import pytest
import pandas as pd
import numpy as np
from src.risk.risk_manager import RiskManager
from unittest.mock import Mock, patch

@pytest.fixture
def risk_manager():
    """RiskManager 인스턴스 생성"""
    config = {
        'initial_capital': 10000.0,
        'position_limits': {
            'max_position_size': 0.1,
            'max_order_size': 0.05
        },
        'risk_limits': {
            'max_drawdown': 0.2,
            'stop_loss': 0.05,
            'take_profit': 0.1,
            'trailing_stop': 0.03,
            'max_concentration': 0.3
        },
        'volatility_limits': {
            'threshold': 0.02
        }
    }
    return RiskManager(config)

def test_calculate_position_size(risk_manager):
    """포지션 크기 계산 테스트"""
    # 정상 케이스
    size, risk_info = risk_manager.calculate_position_size(
        balance=10000.0,
        price=50000.0,
        risk_per_trade=0.02
    )
    assert size > 0
    assert risk_info["risk_limit_exceeded"] is False
    
    # 최소 거래 금액 미달 케이스
    size, risk_info = risk_manager.calculate_position_size(
        balance=100.0,
        price=50000.0,
        risk_per_trade=0.02
    )
    assert size == 0.0
    
    # 가격이 0인 케이스
    size, risk_info = risk_manager.calculate_position_size(
        balance=10000.0,
        price=0.0,
        risk_per_trade=0.02
    )
    assert size == 0.0

def test_calculate_stop_loss(risk_manager):
    """손절가 계산 테스트"""
    # 롱 포지션
    stop_loss = risk_manager.calculate_stop_loss(
        entry_price=50000.0,
        side="long",
        stop_loss_pct=0.02
    )
    assert stop_loss == 49000.0
    
    # 숏 포지션
    stop_loss = risk_manager.calculate_stop_loss(
        entry_price=50000.0,
        side="short",
        stop_loss_pct=0.02
    )
    assert stop_loss == 51000.0
    
    # 잘못된 포지션 방향
    with pytest.raises(ValueError):
        risk_manager.calculate_stop_loss(
            entry_price=50000.0,
            side="invalid",
            stop_loss_pct=0.02
        )

def test_calculate_take_profit(risk_manager):
    """익절가 계산 테스트"""
    # 롱 포지션
    take_profit = risk_manager.calculate_take_profit(
        entry_price=50000.0,
        side="long",
        take_profit_pct=0.04
    )
    assert take_profit == 52000.0
    
    # 숏 포지션
    take_profit = risk_manager.calculate_take_profit(
        entry_price=50000.0,
        side="short",
        take_profit_pct=0.04
    )
    assert take_profit == 48000.0

def test_check_risk_limit(risk_manager):
    """리스크 한도 체크 테스트"""
    # 정상 케이스
    risk_info = risk_manager.check_risk_limit(
        symbol="BTC",
        side="buy",
        price=50000.0,
        size=0.01,
        current_capital=10000.0
    )
    assert risk_info["risk_limit_exceeded"] is False
    
    # 포지션 크기 한도 초과
    risk_info = risk_manager.check_risk_limit(
        symbol="BTC",
        side="buy",
        price=50000.0,
        size=1.0,
        current_capital=10000.0
    )
    assert risk_info["risk_limit_exceeded"] is True

def test_calculate_position_risk(risk_manager):
    """포지션 리스크 계산 테스트"""
    positions = {
        "BTC": {
            "size": 0.01,
            "entry_price": 50000.0,
            "side": "long"
        }
    }
    
    risk_info = risk_manager.calculate_position_risk(
        positions=positions,
        current_capital=10000.0
    )
    
    assert "risk_score" in risk_info
    assert "total_position_ratio" in risk_info
    assert "max_position_ratio" in risk_info
    assert "daily_loss" in risk_info

def test_calculate_concentration_risk(risk_manager):
    """집중도 리스크 계산 테스트"""
    positions = {
        "BTC": {
            "size": 0.01,
            "entry_price": 50000.0
        },
        "ETH": {
            "size": 0.1,
            "entry_price": 3000.0
        }
    }
    
    risk_info = risk_manager.calculate_concentration_risk(
        positions=positions,
        total_capital=10000.0
    )
    
    assert "concentration_score" in risk_info
    assert "max_position_ratio" in risk_info
    assert "herfindahl_index" in risk_info
    assert "risk_level" in risk_info

def test_calculate_volatility_risk(risk_manager):
    """변동성 리스크 계산 테스트"""
    market_data = {
        "BTC": {
            "price": 50000.0,
            "volume": 1000.0,
            "volatility": 0.02
        }
    }
    
    risk_info = risk_manager.calculate_volatility_risk(
        market_data=market_data
    )
    
    assert "risk_score" in risk_info
    assert "volatility" in risk_info
    assert "daily_loss" in risk_info

def test_calculate_liquidity_risk(risk_manager):
    """유동성 리스크 계산 테스트"""
    market_data = {
        "BTC": {
            "price": 50000.0,
            "volume": 1000.0,
            "bid_ask_spread": 10.0
        }
    }
    
    risk_info = risk_manager.calculate_liquidity_risk(
        market_data=market_data
    )
    
    assert "risk_score" in risk_info
    assert "liquidity" in risk_info
    assert "daily_volume" in risk_info

def test_calculate_concentration_risk(risk_manager):
    """포지션 집중 리스크 계산 테스트"""
    # 정상 케이스 (균형 잡힌 포지션)
    positions = {
        'BTC': {'size': 0.5, 'price': 50000},
        'ETH': {'size': 1.0, 'price': 3000},
        'SOL': {'size': 10.0, 'price': 100}
    }
    risk_metrics = risk_manager.calculate_concentration_risk(
        positions=positions,
        total_capital=100000
    )
    assert risk_metrics['concentration_score'] < 0.4
    assert risk_metrics['risk_level'] == 'low'
    
    # 높은 집중도 케이스
    positions = {
        'BTC': {'size': 1.0, 'price': 50000},
        'ETH': {'size': 0.1, 'price': 3000},
        'SOL': {'size': 0.1, 'price': 100}
    }
    risk_metrics = risk_manager.calculate_concentration_risk(
        positions=positions,
        total_capital=100000
    )
    assert risk_metrics['concentration_score'] > 0.7
    assert risk_metrics['risk_level'] == 'high'
    
    # 빈 포지션 케이스
    risk_metrics = risk_manager.calculate_concentration_risk(
        positions={},
        total_capital=100000
    )
    assert risk_metrics['concentration_score'] == 0.0
    assert risk_metrics['risk_level'] == 'low'

def test_adjust_for_concentration_risk(risk_manager):
    """집중 리스크에 따른 포지션 조정 테스트"""
    # 정상 케이스 (조정 불필요)
    positions = {
        'BTC': {'size': 0.5, 'price': 50000},
        'ETH': {'size': 1.0, 'price': 3000},
        'SOL': {'size': 10.0, 'price': 100}
    }
    adjusted_positions = risk_manager.adjust_for_concentration_risk(
        positions=positions,
        total_capital=100000,
        max_concentration=0.3
    )
    assert adjusted_positions['BTC'] == 0.5
    assert adjusted_positions['ETH'] == 1.0
    assert adjusted_positions['SOL'] == 10.0
    
    # 높은 집중도 케이스 (조정 필요)
    positions = {
        'BTC': {'size': 1.0, 'price': 50000},
        'ETH': {'size': 0.1, 'price': 3000},
        'SOL': {'size': 0.1, 'price': 100}
    }
    adjusted_positions = risk_manager.adjust_for_concentration_risk(
        positions=positions,
        total_capital=100000,
        max_concentration=0.3
    )
    assert adjusted_positions['BTC'] < 1.0
    assert adjusted_positions['ETH'] < 0.1
    assert adjusted_positions['SOL'] < 0.1

def test_empty_positions_concentration_risk(risk_manager):
    result = risk_manager.calculate_concentration_risk({}, 100000.0)
    assert result['concentration_score'] == 0.0
    assert result['risk_level'] == 'low'

def test_adjust_for_concentration_risk(risk_manager):
    # 높은 집중도의 포지션
    positions = {
        'BTC': {'size': 2.0, 'price': 50000.0},  # 100,000 USD
        'ETH': {'size': 1.0, 'price': 3000.0},   # 3,000 USD
    }
    total_capital = 150000.0
    
    # 포지션 조정
    adjusted = risk_manager.adjust_for_concentration_risk(
        positions, 
        total_capital,
        max_concentration=0.3
    )
    
    # 결과 검증
    assert isinstance(adjusted, dict)
    assert all(isinstance(size, float) for size in adjusted.values())
    assert all(size > 0 for size in adjusted.values())
    assert 'BTC' in adjusted
    assert 'ETH' in adjusted
    # BTC 포지션이 조정되었는지 확인
    assert adjusted['BTC'] < positions['BTC']['size'] 