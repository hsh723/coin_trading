import pytest
from src.strategy.strategy_manager import StrategyManager
import os
import json
import pandas as pd
import numpy as np

@pytest.fixture
def strategy_manager():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "strategies": {
                "moving_average": {
                    "short_window": 20,
                    "long_window": 50
                },
                "rsi": {
                    "window": 14,
                    "overbought": 70,
                    "oversold": 30
                },
                "macd": {
                    "fast_window": 12,
                    "slow_window": 26,
                    "signal_window": 9
                }
            },
            "risk_management": {
                "max_position_size": 1.0,
                "max_drawdown": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.04
            }
        }
    }
    with open(os.path.join(config_dir, "strategy.json"), "w") as f:
        json.dump(config, f)
    
    return StrategyManager(config_dir=config_dir, data_dir=data_dir)

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

def test_strategy_initialization(strategy_manager):
    assert strategy_manager is not None
    assert strategy_manager.config_dir == "./config"
    assert strategy_manager.data_dir == "./data"

def test_strategy_start_stop(strategy_manager):
    strategy_manager.start()
    assert strategy_manager.is_running() is True
    
    strategy_manager.stop()
    assert strategy_manager.is_running() is False

def test_strategy_registration(strategy_manager):
    strategy_manager.start()
    
    # 전략 등록
    strategy = {
        "name": "test_strategy",
        "type": "technical",
        "parameters": {
            "param1": 1.0,
            "param2": 2.0
        }
    }
    
    result = strategy_manager.register_strategy(strategy)
    assert result is True
    
    # 등록된 전략 확인
    registered_strategy = strategy_manager.get_strategy("test_strategy")
    assert registered_strategy is not None
    assert registered_strategy["name"] == "test_strategy"
    assert registered_strategy["type"] == "technical"
    
    strategy_manager.stop()

def test_strategy_removal(strategy_manager):
    strategy_manager.start()
    
    # 전략 등록 후 제거
    strategy = {
        "name": "test_strategy",
        "type": "technical",
        "parameters": {
            "param1": 1.0,
            "param2": 2.0
        }
    }
    
    strategy_manager.register_strategy(strategy)
    result = strategy_manager.remove_strategy("test_strategy")
    assert result is True
    
    # 제거 확인
    assert strategy_manager.get_strategy("test_strategy") is None
    
    strategy_manager.stop()

def test_strategy_parameter_update(strategy_manager):
    strategy_manager.start()
    
    # 전략 등록 후 파라미터 업데이트
    strategy = {
        "name": "test_strategy",
        "type": "technical",
        "parameters": {
            "param1": 1.0,
            "param2": 2.0
        }
    }
    
    strategy_manager.register_strategy(strategy)
    
    new_parameters = {
        "param1": 2.0,
        "param2": 3.0
    }
    
    result = strategy_manager.update_strategy_parameters(
        "test_strategy",
        new_parameters
    )
    
    assert result is True
    
    # 업데이트 확인
    updated_strategy = strategy_manager.get_strategy("test_strategy")
    assert updated_strategy["parameters"]["param1"] == 2.0
    assert updated_strategy["parameters"]["param2"] == 3.0
    
    strategy_manager.stop()

def test_strategy_signal_generation(strategy_manager, sample_data):
    strategy_manager.start()
    
    # 전략 등록
    strategy = {
        "name": "test_strategy",
        "type": "technical",
        "parameters": {
            "param1": 1.0,
            "param2": 2.0
        }
    }
    
    strategy_manager.register_strategy(strategy)
    
    # 신호 생성
    signals = strategy_manager.generate_signals(
        strategy_name="test_strategy",
        data=sample_data
    )
    
    assert signals is not None
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(sample_data)
    assert all(signals.isin([-1, 0, 1]))
    
    strategy_manager.stop()

def test_strategy_performance(strategy_manager, sample_data):
    strategy_manager.start()
    
    # 전략 등록
    strategy = {
        "name": "test_strategy",
        "type": "technical",
        "parameters": {
            "param1": 1.0,
            "param2": 2.0
        }
    }
    
    strategy_manager.register_strategy(strategy)
    
    # 성과 분석
    performance = strategy_manager.analyze_strategy_performance(
        strategy_name="test_strategy",
        data=sample_data
    )
    
    assert performance is not None
    assert "total_return" in performance
    assert "sharpe_ratio" in performance
    assert "max_drawdown" in performance
    assert "win_rate" in performance
    
    strategy_manager.stop()

def test_strategy_optimization(strategy_manager, sample_data):
    strategy_manager.start()
    
    # 전략 등록
    strategy = {
        "name": "test_strategy",
        "type": "technical",
        "parameters": {
            "param1": 1.0,
            "param2": 2.0
        }
    }
    
    strategy_manager.register_strategy(strategy)
    
    # 최적화
    optimized_params = strategy_manager.optimize_strategy(
        strategy_name="test_strategy",
        data=sample_data,
        param_ranges={
            "param1": (0.5, 2.0),
            "param2": (1.0, 3.0)
        }
    )
    
    assert optimized_params is not None
    assert "param1" in optimized_params
    assert "param2" in optimized_params
    assert "performance" in optimized_params
    
    strategy_manager.stop()

def test_error_handling(strategy_manager):
    strategy_manager.start()
    
    # 잘못된 전략 등록 시도
    with pytest.raises(Exception):
        strategy_manager.register_strategy({
            "name": "test_strategy",
            "type": "invalid_type",
            "parameters": {}
        })
    
    # 에러 통계 확인
    error_stats = strategy_manager.get_error_stats()
    assert error_stats is not None
    assert error_stats["error_count"] > 0
    
    strategy_manager.stop() 