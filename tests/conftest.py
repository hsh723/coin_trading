import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """테스트용 시장 데이터 생성"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(100, 5, len(dates)),
        'high': np.random.normal(105, 5, len(dates)),
        'low': np.random.normal(95, 5, len(dates)),
        'close': np.random.normal(100, 5, len(dates)),
        'volume': np.random.normal(1000, 100, len(dates))
    })
    
    return data

@pytest.fixture
def mock_binance_api() -> Dict[str, Any]:
    """바이낸스 API 모킹"""
    return {
        'fetch_ohlcv': lambda symbol, timeframe, since, limit: [
            [int(datetime.now().timestamp() * 1000), 100, 105, 95, 100, 1000]
        ],
        'fetch_ticker': lambda symbol: {
            'symbol': symbol,
            'last': 100,
            'bid': 99,
            'ask': 101,
            'volume': 1000
        }
    }

@pytest.fixture
def sample_strategy_params() -> Dict[str, Any]:
    """전략 파라미터 테스트용"""
    return {
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }

@pytest.fixture
def sample_risk_params() -> Dict[str, Any]:
    """리스크 관리 파라미터 테스트용"""
    return {
        'max_position_size': 0.1,
        'max_drawdown': 0.2,
        'stop_loss': 0.05,
        'take_profit': 0.1,
        'trailing_stop': 0.03
    }

@pytest.fixture
def sample_trade() -> Dict[str, Any]:
    """거래 데이터 테스트용"""
    return {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'amount': 0.1,
        'price': 100,
        'timestamp': int(datetime.now().timestamp() * 1000)
    } 