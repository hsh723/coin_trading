"""
트레이딩 시스템 통합 테스트
"""

import unittest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.trading.strategy import IntegratedStrategy
from src.trading.execution import OrderExecutor
from src.trading.risk import RiskManager
from src.analysis.performance import PerformanceAnalyzer
from src.notification.telegram_bot import TelegramNotifier
from src.utils.config_loader import get_config

class TestTradingSystem(unittest.TestCase):
    """트레이딩 시스템 통합 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 설정"""
        # 설정 로드
        cls.config = get_config()
        
        # 컴포넌트 초기화
        cls.strategy = IntegratedStrategy()
        cls.executor = OrderExecutor(
            exchange=cls.config['exchange']['name'],
            testnet=True,
            initial_capital=cls.config['trading']['initial_capital']
        )
        cls.risk_manager = RiskManager(
            initial_capital=cls.config['trading']['initial_capital'],
            daily_loss_limit=cls.config['trading']['risk']['daily_loss_limit'],
            weekly_loss_limit=cls.config['trading']['risk']['weekly_loss_limit'],
            monthly_loss_limit=cls.config['trading']['risk']['monthly_loss_limit'],
            max_drawdown_limit=cls.config['trading']['risk']['max_drawdown_limit'],
            max_positions=cls.config['trading']['risk']['max_positions'],
            max_position_size=cls.config['trading']['risk']['max_position_size'],
            max_exposure=cls.config['trading']['risk']['max_exposure'],
            volatility_window=cls.config['trading']['risk']['volatility_window'],
            volatility_threshold=cls.config['trading']['risk']['volatility_threshold'],
            trailing_stop_activation=cls.config['trading']['risk']['trailing_stop_activation'],
            trailing_stop_distance=cls.config['trading']['risk']['trailing_stop_distance']
        )
        cls.analyzer = PerformanceAnalyzer()
        cls.notifier = TelegramNotifier(
            config=cls.config,
            bot_token=cls.config['telegram']['bot_token'],
            chat_id=cls.config['telegram']['chat_id']
        )
    
    async def async_setUp(self):
        """비동기 테스트 설정"""
        # 컴포넌트 초기화
        await self.executor.initialize()
        await self.notifier.initialize()
    
    async def async_tearDown(self):
        """비동기 테스트 정리"""
        # 컴포넌트 정리
        await self.executor.close()
        await self.notifier.close()
    
    def setUp(self):
        """테스트 설정"""
        # 이벤트 루프 생성
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # 비동기 설정 실행
        self.loop.run_until_complete(self.async_setUp())
    
    def tearDown(self):
        """테스트 정리"""
        # 비동기 정리 실행
        self.loop.run_until_complete(self.async_tearDown())
        
        # 이벤트 루프 정리
        self.loop.close()
    
    def test_trading_flow(self):
        """트레이딩 흐름 테스트"""
        async def run_test():
            # 시장 데이터 가져오기
            symbol = 'BTC/USDT'
            timeframe = '1h'
            start_time = datetime.now() - timedelta(days=1)
            end_time = datetime.now()
            
            market_data = await self.executor.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            # 전략 신호 생성
            signals = self.strategy.generate_signals(market_data)
            
            # 리스크 체크
            risk_status = self.risk_manager.check_trading_status()
            self.assertTrue(risk_status['can_trade'])
            
            # 포지션 크기 계산
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                price=market_data['close'].iloc[-1],
                volatility=market_data['close'].pct_change().std()
            )
            
            # 주문 실행
            order = await self.executor.place_order(
                symbol=symbol,
                side='buy',
                amount=position_size,
                price=market_data['close'].iloc[-1]
            )
            
            # 주문 상태 확인
            self.assertEqual(order['status'], 'filled')
            
            # 포지션 업데이트
            positions = await self.executor.get_positions()
            self.assertIn(symbol, positions)
            
            # 성과 분석
            self.analyzer.add_trade({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'buy',
                'price': market_data['close'].iloc[-1],
                'amount': position_size,
                'pnl': 0.0,
                'fee': order['fee']
            })
            
            # 알림 전송
            await self.notifier.send_message(
                f"🔔 거래 실행\n"
                f"심볼: {symbol}\n"
                f"방향: 매수\n"
                f"가격: {market_data['close'].iloc[-1]:,.2f}\n"
                f"수량: {position_size:.4f}"
            )
        
        # 테스트 실행
        self.loop.run_until_complete(run_test())
    
    def test_risk_management(self):
        """리스크 관리 테스트"""
        async def run_test():
            # 여러 거래 실행
            symbol = 'BTC/USDT'
            price = 50000.0
            
            for _ in range(3):
                # 리스크 체크
                risk_status = self.risk_manager.check_trading_status()
                self.assertTrue(risk_status['can_trade'])
                
                # 포지션 크기 계산
                position_size = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    price=price,
                    volatility=0.01
                )
                
                # 주문 실행
                order = await self.executor.place_order(
                    symbol=symbol,
                    side='buy',
                    amount=position_size,
                    price=price
                )
                
                # 포지션 업데이트
                positions = await self.executor.get_positions()
                self.assertLessEqual(len(positions), self.risk_manager.max_positions)
                
                # 성과 분석
                self.analyzer.add_trade({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': 'buy',
                    'price': price,
                    'amount': position_size,
                    'pnl': 0.0,
                    'fee': order['fee']
                })
            
            # 최대 포지션 수 확인
            positions = await self.executor.get_positions()
            self.assertLessEqual(len(positions), self.risk_manager.max_positions)
        
        # 테스트 실행
        self.loop.run_until_complete(run_test())
    
    def test_performance_analysis(self):
        """성과 분석 테스트"""
        async def run_test():
            # 거래 데이터 생성
            trades = [
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
            for trade in trades:
                self.analyzer.add_trade(trade)
            
            # 성과 지표 계산
            metrics = self.analyzer.calculate_metrics()
            
            # 지표 확인
            self.assertGreater(metrics.total_trades, 0)
            self.assertGreater(metrics.winning_trades, 0)
            self.assertGreater(metrics.losing_trades, 0)
            self.assertNotEqual(metrics.total_pnl, 0)
            self.assertGreater(metrics.win_rate, 0)
            self.assertGreater(metrics.profit_factor, 0)
            
            # 포지션별 성과 분석
            for symbol in ['BTC/USDT', 'ETH/USDT']:
                position_metrics = self.analyzer.calculate_position_metrics(symbol)
                self.assertIsNotNone(position_metrics)
                self.assertGreater(position_metrics.total_trades, 0)
        
        # 테스트 실행
        self.loop.run_until_complete(run_test())
    
    def test_notification_system(self):
        """알림 시스템 테스트"""
        async def run_test():
            # 거래 알림
            await self.notifier.send_message(
                f"🔔 거래 알림\n"
                f"심볼: BTC/USDT\n"
                f"방향: 매수\n"
                f"가격: 50000.0\n"
                f"수량: 0.1"
            )
            
            # 오류 알림
            await self.notifier.send_message(
                f"❌ 오류 알림\n"
                f"유형: API 오류\n"
                f"메시지: 연결 실패"
            )
            
            # 성과 알림
            await self.notifier.send_message(
                f"📊 일일 성과\n"
                f"총 수익률: +1.5%\n"
                f"거래 수: 5\n"
                f"승률: 60%"
            )
        
        # 테스트 실행
        self.loop.run_until_complete(run_test())

if __name__ == '__main__':
    unittest.main() 