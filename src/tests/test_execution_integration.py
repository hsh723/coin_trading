"""
실행 시스템 통합 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List

from src.execution.manager import ExecutionManager
from src.trading.execution import OrderExecutor
from src.trading.risk_manager import RiskManager
from src.real_time.processor.real_time_manager import RealTimeManager
from src.monitoring.monitoring_system import MonitoringSystem
from src.notification.telegram import TelegramNotifier

@pytest.fixture
async def trading_system():
    """거래 시스템 픽스처"""
    # 설정
    config = {
        'exchange': {
            'name': 'binance',
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'testnet': True
        },
        'execution': {
            'default_strategy': 'adaptive',
            'risk_limit': 0.02,
            'max_slippage': 0.003,
            'execution_timeout': 300,
            'retry_limit': 3,
            'concurrent_limit': 5
        },
        'risk': {
            'max_position_size': 1.0,
            'max_leverage': 3.0,
            'max_drawdown': 0.1,
            'stop_loss_threshold': 0.05
        },
        'monitoring': {
            'health_check_interval': 60,
            'performance_window': 3600,
            'alert_threshold': 0.8
        }
    }
    
    # 컴포넌트 초기화
    execution_manager = ExecutionManager(config['execution'])
    order_executor = OrderExecutor(config['exchange'])
    risk_manager = RiskManager(config['risk'])
    real_time_manager = RealTimeManager(config)
    monitoring_system = MonitoringSystem(config['monitoring'])
    notifier = TelegramNotifier(config)
    
    # 시스템 구성
    system = {
        'config': config,
        'execution_manager': execution_manager,
        'order_executor': order_executor,
        'risk_manager': risk_manager,
        'real_time_manager': real_time_manager,
        'monitoring_system': monitoring_system,
        'notifier': notifier
    }
    
    # 초기화
    await order_executor.initialize()
    await real_time_manager.initialize()
    await monitoring_system.initialize()
    
    yield system
    
    # 정리
    await order_executor.close()
    await real_time_manager.stop()
    await monitoring_system.stop()
    
@pytest.fixture
def sample_orders():
    """샘플 주문 목록 픽스처"""
    return [
        {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.1,
            'price': 50000.0,
            'leverage': 1.0,
            'time_window': 3600,
            'strategy': 'twap'
        },
        {
            'symbol': 'ETH/USDT',
            'side': 'sell',
            'amount': 1.0,
            'price': 3000.0,
            'leverage': 2.0,
            'volume_profile': [0.1] * 10,
            'strategy': 'vwap'
        },
        {
            'symbol': 'BNB/USDT',
            'side': 'buy',
            'amount': 5.0,
            'price': 300.0,
            'leverage': 1.0,
            'urgency': 0.7,
            'strategy': 'is'
        }
    ]
    
@pytest.mark.asyncio
async def test_end_to_end_execution(trading_system, sample_orders):
    """종단간 실행 테스트"""
    try:
        execution_results = []
        
        # 시스템 상태 모니터링 시작
        await trading_system['monitoring_system'].start_monitoring()
        
        # 실시간 처리 시작
        await trading_system['real_time_manager'].start()
        
        # 주문 실행
        for order in sample_orders:
            # 위험 검사
            risk_check = await trading_system['risk_manager'].check_order_risk(order)
            assert risk_check['approved'], f"위험 검사 실패: {risk_check['reason']}"
            
            # 시장 데이터 조회
            market_data = await trading_system['order_executor'].get_market_data(
                order['symbol'],
                '1m',
                100
            )
            assert market_data is not None, "시장 데이터 조회 실패"
            
            # 주문 실행
            result = await trading_system['execution_manager'].execute_order(
                order,
                order['strategy'],
                market_data
            )
            execution_results.append(result)
            
            # 실행 결과 검증
            assert result is not None, "실행 결과 누락"
            assert 'execution_id' in result, "실행 ID 누락"
            assert result.get('status') != 'failed', f"실행 실패: {result.get('error')}"
            
            # 성과 지표 검증
            metrics = result.get('performance_metrics', {})
            assert metrics.get('completion_rate', 0) > 0.95, "낮은 완료율"
            assert metrics.get('slippage', 1) < 0.005, "높은 슬리피지"
            
            # 알림 전송
            notification = {
                'type': 'execution_complete',
                'execution_id': result['execution_id'],
                'symbol': order['symbol'],
                'strategy': order['strategy'],
                'metrics': metrics
            }
            await trading_system['notifier'].send_notification(notification)
            
        # 통계 검증
        stats = trading_system['execution_manager'].get_execution_stats()
        assert stats['total_executions'] == len(sample_orders)
        assert stats['success_rate'] > 0.95
        
        # 시스템 상태 검증
        health_status = await trading_system['monitoring_system'].get_system_health()
        assert health_status['status'] == 'healthy'
        
        # 실시간 지표 검증
        real_time_metrics = await trading_system['real_time_manager'].get_performance_metrics()
        assert real_time_metrics['system_load'] < 0.8
        assert real_time_metrics['memory_usage'] < 0.8
        assert real_time_metrics['network_latency'] < 100
        
    except Exception as e:
        await trading_system['notifier'].send_notification({
            'type': 'error',
            'error': str(e),
            'timestamp': datetime.now()
        })
        raise
        
@pytest.mark.asyncio
async def test_concurrent_execution(trading_system, sample_orders):
    """동시 실행 테스트"""
    try:
        # 동시 실행 작업 생성
        tasks = [
            trading_system['execution_manager'].execute_order(
                order,
                order['strategy'],
                await trading_system['order_executor'].get_market_data(
                    order['symbol'],
                    '1m',
                    100
                )
            )
            for order in sample_orders
        ]
        
        # 동시 실행
        results = await asyncio.gather(*tasks)
        
        # 결과 검증
        assert len(results) == len(sample_orders)
        for result in results:
            assert result is not None
            assert result.get('status') != 'failed'
            
        # 시스템 부하 검증
        system_metrics = await trading_system['monitoring_system'].get_system_metrics()
        assert system_metrics['cpu_usage'] < 0.9
        assert system_metrics['memory_usage'] < 0.9
        
    except Exception as e:
        await trading_system['notifier'].send_notification({
            'type': 'error',
            'error': str(e),
            'timestamp': datetime.now()
        })
        raise
        
@pytest.mark.asyncio
async def test_error_handling(trading_system):
    """오류 처리 테스트"""
    # 잘못된 주문
    invalid_order = {
        'symbol': 'INVALID/USDT',
        'side': 'buy',
        'amount': -1.0,
        'strategy': 'invalid'
    }
    
    with pytest.raises(ValueError):
        await trading_system['execution_manager'].execute_order(
            invalid_order,
            'twap',
            None
        )
        
    # 네트워크 오류 시뮬레이션
    original_get_market_data = trading_system['order_executor'].get_market_data
    
    async def failing_get_market_data(*args, **kwargs):
        raise Exception("네트워크 오류")
        
    trading_system['order_executor'].get_market_data = failing_get_market_data
    
    try:
        with pytest.raises(Exception, match="네트워크 오류"):
            await trading_system['execution_manager'].execute_order(
                sample_orders[0],
                'twap',
                None
            )
    finally:
        trading_system['order_executor'].get_market_data = original_get_market_data
        
@pytest.mark.asyncio
async def test_risk_management(trading_system, sample_orders):
    """위험 관리 테스트"""
    # 레버리지 한도 초과
    high_leverage_order = sample_orders[0].copy()
    high_leverage_order['leverage'] = 10.0
    
    risk_check = await trading_system['risk_manager'].check_order_risk(high_leverage_order)
    assert not risk_check['approved']
    assert 'leverage' in risk_check['reason']
    
    # 포지션 크기 한도 초과
    large_position_order = sample_orders[0].copy()
    large_position_order['amount'] = 100.0
    
    risk_check = await trading_system['risk_manager'].check_order_risk(large_position_order)
    assert not risk_check['approved']
    assert 'position_size' in risk_check['reason']
    
    # 정상 주문
    risk_check = await trading_system['risk_manager'].check_order_risk(sample_orders[0])
    assert risk_check['approved']
    
@pytest.mark.asyncio
async def test_monitoring_alerts(trading_system):
    """모니터링 알림 테스트"""
    # 시스템 부하 시뮬레이션
    await trading_system['monitoring_system'].update_system_metrics({
        'cpu_usage': 0.95,
        'memory_usage': 0.9,
        'network_latency': 500
    })
    
    # 알림 검증
    alerts = await trading_system['monitoring_system'].get_active_alerts()
    assert len(alerts) > 0
    assert any(alert['type'] == 'high_system_load' for alert in alerts)
    
    # 알림 전송 검증
    notifications = []
    original_send = trading_system['notifier'].send_notification
    
    async def mock_send(notification):
        notifications.append(notification)
        
    trading_system['notifier'].send_notification = mock_send
    
    try:
        await trading_system['monitoring_system'].process_alerts()
        assert len(notifications) > 0
        assert any(n['type'] == 'alert' for n in notifications)
    finally:
        trading_system['notifier'].send_notification = original_send
        
@pytest.mark.asyncio
async def test_performance_tracking(trading_system, sample_orders):
    """성과 추적 테스트"""
    # 실행 이력 생성
    for order in sample_orders:
        result = await trading_system['execution_manager'].execute_order(
            order,
            order['strategy'],
            await trading_system['order_executor'].get_market_data(
                order['symbol'],
                '1m',
                100
            )
        )
        
        # 성과 지표 업데이트
        await trading_system['monitoring_system'].update_performance_metrics({
            'execution_id': result['execution_id'],
            'metrics': result['performance_metrics']
        })
        
    # 성과 통계 검증
    performance_stats = await trading_system['monitoring_system'].get_performance_stats()
    assert 'average_execution_time' in performance_stats
    assert 'average_slippage' in performance_stats
    assert 'success_rate' in performance_stats
    
    # 성과 이상 감지
    anomalies = await trading_system['monitoring_system'].detect_performance_anomalies()
    assert isinstance(anomalies, list)
    
    # 성과 보고서 생성
    report = await trading_system['monitoring_system'].generate_performance_report()
    assert 'summary' in report
    assert 'details' in report
    assert 'recommendations' in report 