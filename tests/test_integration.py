"""
통합 테스트 모듈
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.data.collector import DataCollector
from src.data.processor import DataProcessor
from src.analysis.indicators.technical import TechnicalIndicators
from src.strategy.momentum import MomentumStrategy
from src.risk.manager import RiskManager
from src.notification.telegram import TelegramNotifier
from src.trading.executor import OrderExecutor
from src.backtest.engine import BacktestEngine
from src.trading.trader import Trader

@pytest.fixture
def mock_exchange():
    """거래소 API 모의 객체 생성"""
    mock = MagicMock()
    mock.fetch_ohlcv.return_value = [
        [1625097600000, 35000, 36000, 34000, 35500, 100],
        [1625097900000, 35500, 36500, 35000, 36000, 120],
        [1625098200000, 36000, 37000, 35500, 36500, 150]
    ]
    mock.create_order.return_value = {
        'id': '123456',
        'symbol': 'BTC/USDT',
        'type': 'market',
        'side': 'buy',
        'price': 35500,
        'amount': 0.1,
        'status': 'closed'
    }
    return mock

@pytest.fixture
def mock_telegram():
    """텔레그램 알림 모의 객체 생성"""
    mock = MagicMock()
    mock.send_message.return_value = True
    return mock

@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터 생성"""
    dates = pd.date_range(start='2021-01-01', periods=100, freq='H')
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'close': np.random.normal(100, 10, 100),
        'volume': np.random.normal(1000, 100, 100)
    }, index=dates)
    return data

def test_data_pipeline(mock_exchange, sample_data):
    """데이터 수집부터 신호 생성까지의 파이프라인 테스트"""
    with patch('ccxt.binance', return_value=mock_exchange):
        # 데이터 수집
        collector = DataCollector(
            exchange='binance',
            symbols=['BTC/USDT'],
            timeframes=['1h']
        )
        data = collector.get_historical_data(
            symbol='BTC/USDT',
            timeframe='1h',
            start_date=datetime(2021, 7, 1),
            end_date=datetime(2021, 7, 2)
        )
        
        # 데이터 전처리
        processor = DataProcessor(symbol='BTC/USDT', timeframe='1h')
        processed_data = processor.process_data(data)
        
        # 지표 계산
        indicators = TechnicalIndicators(processed_data)
        indicators.calculate_rsi(period=14)
        indicators.calculate_bollinger_bands(period=20, std_dev=2)
        
        # 신호 생성
        strategy = MomentumStrategy(
            data=processed_data,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30
        )
        signals = strategy.generate_signals()
        
        # 결과 검증
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(processed_data)
        assert signals.isin([-1, 0, 1]).all()

def test_backtest_system(mock_exchange, sample_data):
    """백테스트 시스템 전체 흐름 테스트"""
    with patch('ccxt.binance', return_value=mock_exchange):
        # 백테스트 엔진 초기화
        engine = BacktestEngine(
            symbol='BTC/USDT',
            timeframe='1h',
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2021, 1, 2),
            initial_capital=10000
        )
        
        # 전략 설정
        strategy = MomentumStrategy(
            data=sample_data,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30
        )
        
        # 백테스트 실행
        results = engine.run(strategy)
        
        # 결과 검증
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'trades' in results

def test_notification_and_execution(mock_exchange, mock_telegram):
    """알림 시스템과 주문 실행 시스템 연동 테스트"""
    with patch('ccxt.binance', return_value=mock_exchange):
        # 주문 실행기 초기화
        executor = OrderExecutor(
            exchange=mock_exchange,
            symbol='BTC/USDT',
            testnet=True
        )
        
        # 텔레그램 알림 설정
        notifier = TelegramNotifier(
            bot_token='test_token',
            chat_id='test_chat_id'
        )
        
        # 주문 실행 및 알림 전송
        order = executor.create_order(
            side='buy',
            amount=0.1,
            order_type='market'
        )
        
        # 주문 결과 알림
        notifier.send_trade_notification(order)
        
        # 결과 검증
        assert order['status'] == 'closed'
        assert mock_telegram.send_message.called

def test_error_recovery(mock_exchange, mock_telegram):
    """오류 상황 복구 테스트"""
    with patch('ccxt.binance', return_value=mock_exchange):
        # API 오류 시뮬레이션
        mock_exchange.fetch_ohlcv.side_effect = Exception("API Error")
        
        # 데이터 수집 시도
        collector = DataCollector(
            exchange='binance',
            symbols=['BTC/USDT'],
            timeframes=['1h']
        )
        
        # 오류 발생 시 재시도
        with pytest.raises(Exception):
            collector.get_historical_data(
                symbol='BTC/USDT',
                timeframe='1h',
                start_date=datetime(2021, 7, 1),
                end_date=datetime(2021, 7, 2)
            )
        
        # 오류 알림 전송
        notifier = TelegramNotifier(
            bot_token='test_token',
            chat_id='test_chat_id'
        )
        notifier.send_error_notification("API Error")
        
        # 알림 전송 확인
        assert mock_telegram.send_message.called

def test_risk_management_integration(mock_exchange, sample_data):
    """리스크 관리 시스템 통합 테스트"""
    with patch('ccxt.binance', return_value=mock_exchange):
        # 리스크 관리자 초기화
        risk_manager = RiskManager(
            initial_capital=10000,
            max_position_size=0.1,
            max_drawdown=0.2,
            daily_loss_limit=0.05
        )
        
        # 전략 초기화
        strategy = MomentumStrategy(
            data=sample_data,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30
        )
        
        # 신호 생성
        signals = strategy.generate_signals()
        
        # 포지션 크기 계산
        position_size = risk_manager.calculate_position_size(
            price=100,
            stop_loss=0.02,
            risk_per_trade=0.01
        )
        
        # 리스크 한도 검사
        assert risk_manager.check_risk_limits()
        
        # 포지션 진입
        risk_manager.open_position(
            symbol='BTC/USDT',
            position_type='long',
            size=position_size,
            entry_price=100,
            stop_loss=98,
            take_profit=104
        )
        
        # 포지션 업데이트
        risk_manager.update_position(
            symbol='BTC/USDT',
            current_price=102,
            unrealized_pnl=20
        )
        
        # 결과 검증
        assert 'BTC/USDT' in risk_manager.positions
        assert risk_manager.positions['BTC/USDT']['unrealized_pnl'] == 20

def test_end_to_end_trading_flow(mock_binance_api):
    """종단간 트레이딩 흐름 테스트"""
    # 데이터 수집
    collector = DataCollector()
    collector.exchange = mock_binance_api
    
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()
    market_data = collector.get_historical_data(
        symbol='BTC/USDT',
        start_time=start_time,
        end_time=end_time,
        interval='1h'
    )
    
    # 데이터 전처리
    processor = DataProcessor()
    processed_data = processor.preprocess_data(market_data)
    processed_data = processor.add_technical_indicators(processed_data)
    
    # 전략 설정
    strategy = MomentumStrategy()
    strategy.initialize(processed_data)
    
    # 리스크 관리 설정
    risk_manager = RiskManager()
    
    # 트레이더 설정
    trader = Trader()
    trader.setup(strategy, risk_manager)
    trader.balance = 10000
    
    # 거래 실행
    for i in range(len(processed_data)):
        current_data = processed_data.iloc[i]
        signals = strategy.generate_signals(current_data)
        
        if signals['buy_signal']:
            trade = {
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'amount': 0.1,
                'price': current_data['close'],
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
            trader.execute_trade(trade)
        
        elif signals['sell_signal'] and trader.position > 0:
            trader.close_position(current_data)
        
        trader.update_state(current_data)
    
    # 성능 지표 확인
    metrics = trader.get_performance_metrics()
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics

def test_backtest_integration(mock_binance_api):
    """백테스팅 통합 테스트"""
    # 데이터 수집 및 전처리
    collector = DataCollector()
    collector.exchange = mock_binance_api
    
    start_time = datetime.now() - timedelta(days=30)
    end_time = datetime.now()
    market_data = collector.get_historical_data(
        symbol='BTC/USDT',
        start_time=start_time,
        end_time=end_time,
        interval='1h'
    )
    
    processor = DataProcessor()
    processed_data = processor.preprocess_data(market_data)
    processed_data = processor.add_technical_indicators(processed_data)
    
    # 백테스팅 엔진 설정
    engine = BacktestEngine()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    engine.setup(strategy, risk_manager)
    
    # 백테스팅 실행
    results = engine.run(processed_data, initial_balance=10000)
    
    # 결과 검증
    assert 'total_return' in results
    assert 'sharpe_ratio' in results
    assert 'max_drawdown' in results
    assert 'trades' in results
    assert len(results['trades']) > 0

def test_risk_management_integration(mock_binance_api):
    """리스크 관리 통합 테스트"""
    # 데이터 수집
    collector = DataCollector()
    collector.exchange = mock_binance_api
    
    market_data = collector.get_historical_data(
        symbol='BTC/USDT',
        start_time=datetime.now() - timedelta(days=7),
        end_time=datetime.now(),
        interval='1h'
    )
    
    # 트레이딩 시스템 설정
    trader = Trader()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    trader.setup(strategy, risk_manager)
    trader.balance = 10000
    
    # 리스크 제한 테스트
    for i in range(len(market_data)):
        current_data = market_data.iloc[i]
        
        # 큰 포지션 시도
        large_trade = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 10,  # 매우 큰 거래량
            'price': current_data['close'],
            'timestamp': int(datetime.now().timestamp() * 1000)
        }
        
        result = trader.execute_trade(large_trade)
        assert result['status'] == 'rejected'
        assert 'risk_limit' in result['reason']
        
        # 정상적인 거래 시도
        normal_trade = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.1,
            'price': current_data['close'],
            'timestamp': int(datetime.now().timestamp() * 1000)
        }
        
        result = trader.execute_trade(normal_trade)
        assert result['status'] == 'executed'
        
        trader.update_state(current_data)
        
        # 낙폭 모니터링
        if trader.calculate_drawdown() > risk_manager.max_drawdown:
            assert trader.position == 0  # 포지션 청산 확인

def test_data_processing_integration(mock_binance_api):
    """데이터 처리 통합 테스트"""
    # 데이터 수집
    collector = DataCollector()
    collector.exchange = mock_binance_api
    
    market_data = collector.get_historical_data(
        symbol='BTC/USDT',
        start_time=datetime.now() - timedelta(days=7),
        end_time=datetime.now(),
        interval='1h'
    )
    
    # 데이터 전처리
    processor = DataProcessor()
    
    # 결측치 처리
    data_with_nan = market_data.copy()
    data_with_nan.iloc[0, 0] = None
    processed_data = processor.preprocess_data(data_with_nan)
    assert not processed_data.isnull().any().any()
    
    # 이상치 처리
    data_with_outlier = market_data.copy()
    data_with_outlier.iloc[0, 0] = 1000000  # 이상치 추가
    processed_data = processor.preprocess_data(data_with_outlier)
    assert processed_data.iloc[0, 0] != 1000000
    
    # 기술적 지표 추가
    processed_data = processor.add_technical_indicators(processed_data)
    assert 'rsi' in processed_data.columns
    assert 'macd' in processed_data.columns
    assert 'signal' in processed_data.columns
    
    # 시간대 리샘플링
    resampled_data = processor.resample_timeframe(
        processed_data,
        source_timeframe='1h',
        target_timeframe='4h'
    )
    assert len(resampled_data) < len(processed_data)
    assert resampled_data.index.freq == '4H'

def test_performance_monitoring(mock_binance_api):
    """성능 모니터링 통합 테스트"""
    # 데이터 수집 및 전처리
    collector = DataCollector()
    collector.exchange = mock_binance_api
    
    market_data = collector.get_historical_data(
        symbol='BTC/USDT',
        start_time=datetime.now() - timedelta(days=30),
        end_time=datetime.now(),
        interval='1h'
    )
    
    processor = DataProcessor()
    processed_data = processor.preprocess_data(market_data)
    processed_data = processor.add_technical_indicators(processed_data)
    
    # 트레이딩 시스템 설정
    trader = Trader()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    trader.setup(strategy, risk_manager)
    trader.balance = 10000
    
    # 거래 실행 및 성능 모니터링
    for i in range(len(processed_data)):
        current_data = processed_data.iloc[i]
        signals = strategy.generate_signals(current_data)
        
        if signals['buy_signal']:
            trade = {
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'amount': 0.1,
                'price': current_data['close'],
                'timestamp': int(datetime.now().timestamp() * 1000)
            }
            trader.execute_trade(trade)
        
        elif signals['sell_signal'] and trader.position > 0:
            trader.close_position(current_data)
        
        trader.update_state(current_data)
        
        # 성능 지표 모니터링
        metrics = trader.get_performance_metrics()
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        
        # 리스크 모니터링
        assert trader.calculate_drawdown() <= risk_manager.max_drawdown
        assert trader.position <= risk_manager.max_position_size * trader.balance / current_data['close'] 