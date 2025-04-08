"""
리스크 관리 시스템 단위 테스트
"""

import unittest
from datetime import datetime, timedelta
from src.trading.risk import RiskManager, MarketCondition, RiskMetrics

class TestRiskManager(unittest.TestCase):
    """RiskManager 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.risk_manager = RiskManager(
            initial_capital=10000.0,
            daily_loss_limit=0.02,  # 2%
            weekly_loss_limit=0.05,  # 5%
            monthly_loss_limit=0.10,  # 10%
            max_drawdown_limit=0.15,  # 15%
            max_positions=5,
            max_position_size=0.10,  # 10%
            max_exposure=0.30,  # 30%
            volatility_window=20,
            volatility_threshold=0.02,  # 2%
            trailing_stop_activation=0.02,  # 2%
            trailing_stop_distance=0.01  # 1%
        )
    
    def test_initialization(self):
        """초기화 테스트"""
        self.assertEqual(self.risk_manager.initial_capital, 10000.0)
        self.assertEqual(self.risk_manager.current_capital, 10000.0)
        self.assertEqual(self.risk_manager.daily_loss_limit, 0.02)
        self.assertEqual(self.risk_manager.weekly_loss_limit, 0.05)
        self.assertEqual(self.risk_manager.monthly_loss_limit, 0.10)
        self.assertEqual(self.risk_manager.max_drawdown_limit, 0.15)
        self.assertEqual(self.risk_manager.max_positions, 5)
        self.assertEqual(self.risk_manager.max_position_size, 0.10)
        self.assertEqual(self.risk_manager.max_exposure, 0.30)
        self.assertEqual(self.risk_manager.volatility_window, 20)
        self.assertEqual(self.risk_manager.volatility_threshold, 0.02)
        self.assertEqual(self.risk_manager.trailing_stop_activation, 0.02)
        self.assertEqual(self.risk_manager.trailing_stop_distance, 0.01)
        
        # 메트릭스 초기화 확인
        self.assertIsInstance(self.risk_manager.metrics, RiskMetrics)
        self.assertEqual(self.risk_manager.metrics.total_trades, 0)
        self.assertEqual(self.risk_manager.metrics.winning_trades, 0)
        self.assertEqual(self.risk_manager.metrics.losing_trades, 0)
        self.assertEqual(self.risk_manager.metrics.total_pnl, 0.0)
        self.assertEqual(self.risk_manager.metrics.max_drawdown, 0.0)
        self.assertEqual(self.risk_manager.metrics.volatility, 0.0)
    
    def test_check_trading_status(self):
        """거래 상태 확인 테스트"""
        # 정상 상태
        status = self.risk_manager.check_trading_status()
        self.assertTrue(status['can_trade'])
        self.assertEqual(status['reason'], '')
        
        # 일일 손실 한도 초과
        self.risk_manager.metrics.daily_pnl = -250.0  # 2.5% 손실
        status = self.risk_manager.check_trading_status()
        self.assertFalse(status['can_trade'])
        self.assertIn('일일 손실 한도', status['reason'])
        
        # 주간 손실 한도 초과
        self.risk_manager.metrics.weekly_pnl = -600.0  # 6% 손실
        status = self.risk_manager.check_trading_status()
        self.assertFalse(status['can_trade'])
        self.assertIn('주간 손실 한도', status['reason'])
        
        # 월간 손실 한도 초과
        self.risk_manager.metrics.monthly_pnl = -1200.0  # 12% 손실
        status = self.risk_manager.check_trading_status()
        self.assertFalse(status['can_trade'])
        self.assertIn('월간 손실 한도', status['reason'])
    
    def test_calculate_position_size(self):
        """포지션 크기 계산 테스트"""
        # 정상 시장 상태
        size = self.risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            price=50000.0,
            volatility=0.01
        )
        self.assertGreater(size, 0)
        self.assertLessEqual(size, self.risk_manager.max_position_size * self.risk_manager.current_capital)
        
        # 변동성 높은 시장 상태
        size = self.risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            price=50000.0,
            volatility=0.05
        )
        self.assertLess(size, self.risk_manager.max_position_size * self.risk_manager.current_capital)
    
    def test_update_trailing_stop(self):
        """트레일링 스탑 업데이트 테스트"""
        # 포지션 추가
        self.risk_manager.positions['BTC/USDT'] = {
            'side': 'long',
            'amount': 0.1,
            'entry_price': 50000.0,
            'current_price': 50000.0,
            'trailing_stop': 49000.0
        }
        
        # 수익 발생 시 트레일링 스탑 업데이트
        self.risk_manager.positions['BTC/USDT']['current_price'] = 52000.0
        self.risk_manager.update_trailing_stop('BTC/USDT')
        self.assertGreater(self.risk_manager.positions['BTC/USDT']['trailing_stop'], 49000.0)
        
        # 손실 발생 시 트레일링 스탑 유지
        self.risk_manager.positions['BTC/USDT']['current_price'] = 51000.0
        old_stop = self.risk_manager.positions['BTC/USDT']['trailing_stop']
        self.risk_manager.update_trailing_stop('BTC/USDT')
        self.assertEqual(self.risk_manager.positions['BTC/USDT']['trailing_stop'], old_stop)
    
    def test_update_risk_metrics(self):
        """리스크 메트릭스 업데이트 테스트"""
        # 거래 추가
        self.risk_manager.update_risk_metrics(
            symbol='BTC/USDT',
            pnl=100.0,
            timestamp=datetime.now()
        )
        
        # 메트릭스 업데이트 확인
        self.assertEqual(self.risk_manager.metrics.total_trades, 1)
        self.assertEqual(self.risk_manager.metrics.winning_trades, 1)
        self.assertEqual(self.risk_manager.metrics.losing_trades, 0)
        self.assertEqual(self.risk_manager.metrics.total_pnl, 100.0)
        
        # 손실 거래 추가
        self.risk_manager.update_risk_metrics(
            symbol='BTC/USDT',
            pnl=-50.0,
            timestamp=datetime.now()
        )
        
        # 메트릭스 업데이트 확인
        self.assertEqual(self.risk_manager.metrics.total_trades, 2)
        self.assertEqual(self.risk_manager.metrics.winning_trades, 1)
        self.assertEqual(self.risk_manager.metrics.losing_trades, 1)
        self.assertEqual(self.risk_manager.metrics.total_pnl, 50.0)
    
    def test_reset_metrics(self):
        """메트릭스 리셋 테스트"""
        # 거래 추가
        self.risk_manager.update_risk_metrics(
            symbol='BTC/USDT',
            pnl=100.0,
            timestamp=datetime.now()
        )
        
        # 일일 메트릭스 리셋
        self.risk_manager.reset_daily_metrics()
        self.assertEqual(self.risk_manager.metrics.daily_pnl, 0.0)
        self.assertEqual(self.risk_manager.metrics.daily_trades, 0)
        
        # 주간 메트릭스 리셋
        self.risk_manager.reset_weekly_metrics()
        self.assertEqual(self.risk_manager.metrics.weekly_pnl, 0.0)
        self.assertEqual(self.risk_manager.metrics.weekly_trades, 0)
        
        # 월간 메트릭스 리셋
        self.risk_manager.reset_monthly_metrics()
        self.assertEqual(self.risk_manager.metrics.monthly_pnl, 0.0)
        self.assertEqual(self.risk_manager.metrics.monthly_trades, 0)
    
    def test_market_condition(self):
        """시장 상태 분류 테스트"""
        # 정상 시장
        condition = self.risk_manager.classify_market_condition(0.01)
        self.assertEqual(condition, MarketCondition.NORMAL)
        
        # 변동성 시장
        condition = self.risk_manager.classify_market_condition(0.03)
        self.assertEqual(condition, MarketCondition.VOLATILE)
        
        # 극단적 시장
        condition = self.risk_manager.classify_market_condition(0.05)
        self.assertEqual(condition, MarketCondition.EXTREME)

if __name__ == '__main__':
    unittest.main() 