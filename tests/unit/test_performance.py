"""
성과 분석 시스템 단위 테스트
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.analysis.performance import PerformanceAnalyzer, PerformanceMetrics, PositionMetrics

class TestPerformanceAnalyzer(unittest.TestCase):
    """PerformanceAnalyzer 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.analyzer = PerformanceAnalyzer()
        
        # 테스트용 거래 데이터 생성
        self.trades = [
            {
                'timestamp': datetime.now() - timedelta(days=3),
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'price': 50000.0,
                'amount': 0.1,
                'pnl': 100.0,
                'fee': 0.1
            },
            {
                'timestamp': datetime.now() - timedelta(days=2),
                'symbol': 'ETH/USDT',
                'side': 'sell',
                'price': 3000.0,
                'amount': 1.0,
                'pnl': -50.0,
                'fee': 0.1
            },
            {
                'timestamp': datetime.now() - timedelta(days=1),
                'symbol': 'BTC/USDT',
                'side': 'sell',
                'price': 51000.0,
                'amount': 0.1,
                'pnl': 50.0,
                'fee': 0.1
            }
        ]
        
        # 거래 데이터 추가
        for trade in self.trades:
            self.analyzer.add_trade(trade)
    
    def test_initialization(self):
        """초기화 테스트"""
        self.assertEqual(len(self.analyzer.trades), 3)
        self.assertEqual(len(self.analyzer.positions), 2)  # BTC/USDT, ETH/USDT
        self.assertEqual(self.analyzer.initial_capital, 10000.0)
        self.assertEqual(self.analyzer.current_capital, 10000.0)
    
    def test_add_trade(self):
        """거래 추가 테스트"""
        # 새로운 거래 추가
        new_trade = {
            'timestamp': datetime.now(),
            'symbol': 'XRP/USDT',
            'side': 'buy',
            'price': 1.0,
            'amount': 100.0,
            'pnl': 0.0,
            'fee': 0.1
        }
        
        self.analyzer.add_trade(new_trade)
        
        # 거래 추가 확인
        self.assertEqual(len(self.analyzer.trades), 4)
        self.assertEqual(len(self.analyzer.positions), 3)  # XRP/USDT 추가
        self.assertEqual(self.analyzer.positions['XRP/USDT']['total_trades'], 1)
    
    def test_calculate_metrics(self):
        """성과 지표 계산 테스트"""
        metrics = self.analyzer.calculate_metrics()
        
        # 기본 지표 확인
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.total_trades, 3)
        self.assertEqual(metrics.winning_trades, 2)
        self.assertEqual(metrics.losing_trades, 1)
        self.assertEqual(metrics.total_pnl, 100.0)  # 100 - 50 + 50
        
        # 수익률 지표 확인
        self.assertGreater(metrics.win_rate, 0.5)
        self.assertGreater(metrics.profit_factor, 1.0)
        self.assertGreater(metrics.sharpe_ratio, 0)
        self.assertGreater(metrics.sortino_ratio, 0)
        self.assertGreater(metrics.calmar_ratio, 0)
    
    def test_calculate_position_metrics(self):
        """포지션별 성과 지표 계산 테스트"""
        # BTC/USDT 포지션 메트릭스
        btc_metrics = self.analyzer.calculate_position_metrics('BTC/USDT')
        
        self.assertIsInstance(btc_metrics, PositionMetrics)
        self.assertEqual(btc_metrics.total_trades, 2)
        self.assertEqual(btc_metrics.winning_trades, 2)
        self.assertEqual(btc_metrics.losing_trades, 0)
        self.assertEqual(btc_metrics.total_pnl, 150.0)  # 100 + 50
        
        # ETH/USDT 포지션 메트릭스
        eth_metrics = self.analyzer.calculate_position_metrics('ETH/USDT')
        
        self.assertIsInstance(eth_metrics, PositionMetrics)
        self.assertEqual(eth_metrics.total_trades, 1)
        self.assertEqual(eth_metrics.winning_trades, 0)
        self.assertEqual(eth_metrics.losing_trades, 1)
        self.assertEqual(eth_metrics.total_pnl, -50.0)
    
    def test_update_equity_curve(self):
        """자본금 곡선 업데이트 테스트"""
        # 자본금 곡선 업데이트
        self.analyzer.update_equity_curve()
        
        # 자본금 곡선 확인
        self.assertIsInstance(self.analyzer.equity_curve, pd.Series)
        self.assertEqual(len(self.analyzer.equity_curve), 4)  # 초기 + 3거래
        
        # 자본금 변화 확인
        self.assertEqual(self.analyzer.equity_curve.iloc[0], 10000.0)  # 초기 자본
        self.assertEqual(self.analyzer.equity_curve.iloc[-1], 10100.0)  # 최종 자본
    
    def test_calculate_drawdowns(self):
        """낙폭 계산 테스트"""
        # 자본금 곡선 업데이트
        self.analyzer.update_equity_curve()
        
        # 낙폭 계산
        drawdowns = self.analyzer.calculate_drawdowns()
        
        # 낙폭 확인
        self.assertIsInstance(drawdowns, pd.Series)
        self.assertEqual(len(drawdowns), len(self.analyzer.equity_curve))
        self.assertLessEqual(drawdowns.max(), 0)  # 낙폭은 음수여야 함
    
    def test_calculate_returns(self):
        """수익률 계산 테스트"""
        # 수익률 계산
        returns = self.analyzer.calculate_returns()
        
        # 수익률 확인
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), len(self.analyzer.trades))
        
        # 수익률 값 확인
        self.assertEqual(returns.iloc[0], 0.01)  # 100/10000
        self.assertEqual(returns.iloc[1], -0.005)  # -50/10000
        self.assertEqual(returns.iloc[2], 0.005)  # 50/10000
    
    def test_calculate_volatility(self):
        """변동성 계산 테스트"""
        # 수익률 계산
        returns = self.analyzer.calculate_returns()
        
        # 변동성 계산
        volatility = self.analyzer.calculate_volatility(returns)
        
        # 변동성 확인
        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0)
    
    def test_calculate_risk_metrics(self):
        """리스크 지표 계산 테스트"""
        # 수익률 계산
        returns = self.analyzer.calculate_returns()
        
        # 리스크 지표 계산
        risk_metrics = self.analyzer.calculate_risk_metrics(returns)
        
        # 리스크 지표 확인
        self.assertIsInstance(risk_metrics, dict)
        self.assertIn('sharpe_ratio', risk_metrics)
        self.assertIn('sortino_ratio', risk_metrics)
        self.assertIn('calmar_ratio', risk_metrics)
        self.assertIn('max_drawdown', risk_metrics)
        
        # 지표 값 확인
        self.assertGreater(risk_metrics['sharpe_ratio'], 0)
        self.assertGreater(risk_metrics['sortino_ratio'], 0)
        self.assertGreater(risk_metrics['calmar_ratio'], 0)
        self.assertLess(risk_metrics['max_drawdown'], 0)

if __name__ == '__main__':
    unittest.main() 