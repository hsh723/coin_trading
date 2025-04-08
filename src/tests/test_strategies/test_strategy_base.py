import pytest
import pandas as pd
from src.strategy.base import BaseStrategy
from src.risk.manager import RiskManager

class TestBaseStrategy:
    @pytest.fixture
    def strategy(self):
        risk_manager = RiskManager({'max_position_size': 0.1})
        return BaseStrategy(risk_manager)

    def test_signal_generation(self, strategy, sample_data):
        """신호 생성 테스트"""
        signals = strategy.generate_signals(sample_data)
        assert isinstance(signals, dict)
        assert 'action' in signals
