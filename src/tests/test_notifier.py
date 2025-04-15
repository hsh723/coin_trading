"""
실행 시스템 알림 테스트
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from src.execution.notifier import ExecutionNotifier

@pytest.fixture
def notifier():
    """알림 시스템 픽스처"""
    config = {
        'notification': {
            'enabled': True,
            'types': ['telegram'],
            'telegram': {
                'token': 'test_token',
                'chat_id': 'test_chat_id'
            },
            'critical_threshold': 0.9,
            'warning_threshold': 0.7,
            'info_threshold': 0.5,
            'max_history_size': 100,
            'rate_limit': 5,
            'rate_window': 1
        }
    }
    notifier = ExecutionNotifier(config)
    return notifier

@pytest.fixture
async def initialized_notifier(notifier):
    """초기화된 알림 시스템 픽스처"""
    await notifier.initialize()
    yield notifier
    await notifier.close()

@pytest.mark.asyncio
async def test_notifier_initialization(notifier):
    """초기화 테스트"""
    # 초기화
    await notifier.initialize()
    
    # 설정 확인
    assert notifier.enabled
    assert 'telegram' in notifier.notification_types
    assert notifier.telegram is not None
    
    # 정리
    await notifier.close()

@pytest.mark.asyncio
async def test_execution_notification(initialized_notifier):
    """실행 알림 테스트"""
    # 실행 데이터
    execution_data = {
        'order_id': 'test_order',
        'success': True,
        'price': 100.0,
        'volume': 1.0,
        'slippage': 0.001
    }
    
    # 알림 전송
    await initialized_notifier.notify_execution(execution_data)
    
    # 알림 기록 확인
    history = initialized_notifier.get_notification_history()
    assert len(history) == 1
    assert history[0]['level'] == 'info'
    assert 'test_order' in history[0]['message']

@pytest.mark.asyncio
async def test_error_notification(initialized_notifier):
    """오류 알림 테스트"""
    # 오류 데이터
    error_data = {
        'message': 'Test error',
        'type': 'TestError',
        'details': {'reason': 'test'}
    }
    
    # 알림 전송
    await initialized_notifier.notify_error(error_data)
    
    # 알림 기록 확인
    history = initialized_notifier.get_notification_history()
    assert len(history) == 1
    assert history[0]['level'] == 'critical'
    assert 'Test error' in history[0]['message']

@pytest.mark.asyncio
async def test_performance_notification(initialized_notifier):
    """성능 알림 테스트"""
    # 성능 데이터
    performance_data = {
        'latency': 100.0,
        'success_rate': 0.95,
        'slippage': 0.001
    }
    
    # 알림 전송
    await initialized_notifier.notify_performance(performance_data)
    
    # 알림 기록 확인
    history = initialized_notifier.get_notification_history()
    assert len(history) == 1
    assert history[0]['level'] == 'info'
    assert '100.0ms' in history[0]['message']

@pytest.mark.asyncio
async def test_notification_levels(initialized_notifier):
    """알림 레벨 테스트"""
    # 정상 실행
    await initialized_notifier.notify_execution({
        'success': True,
        'slippage': 0.001
    })
    
    # 경고 수준 실행
    await initialized_notifier.notify_execution({
        'success': True,
        'slippage': 0.8
    })
    
    # 심각 수준 실행
    await initialized_notifier.notify_execution({
        'success': True,
        'slippage': 0.95
    })
    
    # 알림 기록 확인
    history = initialized_notifier.get_notification_history()
    assert len(history) == 3
    assert history[0]['level'] == 'info'
    assert history[1]['level'] == 'warning'
    assert history[2]['level'] == 'critical'

@pytest.mark.asyncio
async def test_rate_limit(initialized_notifier):
    """속도 제한 테스트"""
    # 속도 제한 초과
    for i in range(10):
        await initialized_notifier.notify_execution({
            'order_id': f'test_order_{i}',
            'success': True
        })
        
    # 알림 기록 확인
    history = initialized_notifier.get_notification_history()
    assert len(history) == 5  # rate_limit 설정값
    
    # 시간 경과 후 재시도
    await asyncio.sleep(1.1)  # rate_window 초과
    await initialized_notifier.notify_execution({
        'order_id': 'test_order_new',
        'success': True
    })
    
    history = initialized_notifier.get_notification_history()
    assert len(history) == 6

@pytest.mark.asyncio
async def test_history_size_limit(initialized_notifier):
    """기록 크기 제한 테스트"""
    # 최대 크기 초과
    for i in range(150):  # max_history_size = 100
        await initialized_notifier.notify_execution({
            'order_id': f'test_order_{i}',
            'success': True
        })
        await asyncio.sleep(0.1)  # 속도 제한 회피
        
    # 기록 크기 확인
    history = initialized_notifier.get_notification_history()
    assert len(history) == 100
    assert history[-1]['message'].find('test_order_149') != -1

@pytest.mark.asyncio
async def test_notification_filtering(initialized_notifier):
    """알림 기록 필터링 테스트"""
    # 여러 알림 전송
    start_time = datetime.now()
    
    await initialized_notifier.notify_execution({
        'success': True,
        'slippage': 0.001
    })
    
    await initialized_notifier.notify_error({
        'message': 'Test error'
    })
    
    await initialized_notifier.notify_performance({
        'latency': 1500.0,
        'success_rate': 0.8
    })
    
    end_time = datetime.now()
    
    # 시간 범위로 필터링
    history = initialized_notifier.get_notification_history(
        start_time=start_time,
        end_time=end_time
    )
    assert len(history) == 3
    
    # 레벨로 필터링
    critical_history = initialized_notifier.get_notification_history(
        level='critical'
    )
    assert len(critical_history) == 2  # 오류와 높은 지연시간

@pytest.mark.asyncio
async def test_disabled_notification(initialized_notifier):
    """비활성화된 알림 테스트"""
    # 알림 비활성화
    initialized_notifier.enabled = False
    
    # 알림 전송 시도
    await initialized_notifier.notify_execution({
        'success': True
    })
    
    # 알림 기록 확인
    history = initialized_notifier.get_notification_history()
    assert len(history) == 0

@pytest.mark.asyncio
async def test_error_handling(initialized_notifier):
    """오류 처리 테스트"""
    # 잘못된 실행 데이터
    execution_data = {
        'order_id': None,
        'success': 'invalid'
    }
    
    # 오류가 발생해도 계속 실행
    await initialized_notifier.notify_execution(execution_data)
    
    # 알림 시스템이 여전히 동작
    await initialized_notifier.notify_execution({
        'success': True
    })
    
    history = initialized_notifier.get_notification_history()
    assert len(history) > 0

@pytest.mark.asyncio
async def test_concurrent_notifications(initialized_notifier):
    """동시 알림 테스트"""
    async def concurrent_operation(i: int):
        await initialized_notifier.notify_execution({
            'order_id': f'test_order_{i}',
            'success': True
        })
        
    # 여러 알림 동시 전송
    tasks = [concurrent_operation(i) for i in range(5)]
    await asyncio.gather(*tasks)
    
    # 알림 기록 확인
    history = initialized_notifier.get_notification_history()
    assert len(history) == 5 