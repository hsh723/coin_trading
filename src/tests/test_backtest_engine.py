import pytest
import pandas as pd
import numpy as np
from src.backtest.engine import BacktestEngine
from src.strategy.base import BaseStrategy

class TestBacktestEngine:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'open': np.random.random(100),
            'high': np.random.random(100),
            'low': np.random.random(100),
            'close': np.random.random(100),
            'volume': np.random.random(100)
        })
        
    def test_backtest_execution(self, sample_data):
        """백테스트 실행 테스트"""
        engine = BacktestEngine(strategy=MockStrategy(), initial_capital=10000)
        results = engine.run(sample_data)
        
        assert 'trades' in results
        assert 'metrics' in results
        assert results['final_capital'] > 0
