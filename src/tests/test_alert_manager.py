"""
알림 관리자 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from src.execution.alert_manager import AlertManager

@pytest.fixture
def config():
    """테스트 설정"""
    return {
        'alerts': {
            'enabled': True,
            'rate_limit': 1,  # 1초
            'cooldown': 5,  # 5초
            'latency_threshold': 1000,  # ms
            'error_rate_threshold': 0.05,  # 5%
            'fill_rate_threshold': 0.95,  # 95%
            'slippage_threshold': 0.001,  # 0.1%
            'volume_threshold': 100.0
        },
        'telegram': {
            'token': 'test_token',
            'chat_id': 'test_chat_id'
        }
    }

@pytest.fixture
async def alert_manager(config):
    """알림 관리자 인스턴스"""
    manager = AlertManager(config)
    await manager.initialize()
    yield manager
    await manager.close()

@pytest.mark.asyncio
async def test_initialization(config):
    """초기화 테스트"""
    manager = AlertManager(config)
    await manager.initialize()
    
    # 설정 확인
    assert manager.enabled == True
    assert manager.rate_limit == 1
    assert manager.cooldown == 5
    assert manager.thresholds['latency'] == 1000
    assert manager.thresholds['error_rate'] == 0.05
    assert manager.thresholds['fill_rate'] == 0.95
    assert manager.thresholds['slippage'] == 0.001
    assert manager.thresholds['volume'] == 100.0
    
    await manager.close()

@pytest.mark.asyncio
async def test_alert_level_determination(alert_manager):
    """알림 레벨 결정 테스트"""
    # 정상 메트릭
    normal_metrics = {
        'latency': 500,
        'error_rate': 0.01,
        'fill_rate': 0.98,
        'slippage': 0.0005,
        'throughput': 50
    }
    assert alert_manager._determine_alert_level(normal_metrics) == 'normal'
    
    # 경고 메트릭
    warning_metrics = {
        'latency': 1200,
        'error_rate': 0.03,
        'fill_rate': 0.94,
        'slippage': 0.002,
        'throughput': 30
    }
    assert alert_manager._determine_alert_level(warning_metrics) == 'warning'
    
    # 심각 메트릭
    critical_metrics = {
        'latency': 2500,
        'error_rate': 0.15,
        'fill_rate': 0.4,
        'slippage': 0.005,
        'throughput': 10
    }
    assert alert_manager._determine_alert_level(critical_metrics) == 'critical'

@pytest.mark.asyncio
async def test_alert_message_creation(alert_manager):
    """알림 메시지 생성 테스트"""
    # 테스트 메트릭
    metrics = {
        'latency': 1500,
        'error_rate': 0.08,
        'fill_rate': 0.92,
        'slippage': 0.002,
        'throughput': 25
    }
    
    # 경고 메시지
    warning_message = alert_manager._create_alert_message(metrics, 'warning')
    assert '⚠️ 경고' in warning_message
    assert '1500.00ms' in warning_message
    assert '8.00%' in warning_message
    assert '92.00%' in warning_message
    assert '0.2000%' in warning_message
    assert '25.00 TPS' in warning_message
    
    # 심각 메시지
    critical_message = alert_manager._create_alert_message(metrics, 'critical')
    assert '🚨 심각' in critical_message

@pytest.mark.asyncio
async def test_rate_limiting(alert_manager):
    """알림 제한 테스트"""
    # 첫 번째 알림
    assert alert_manager._check_rate_limit('warning') == True
    alert_manager.last_alert_time['warning'] = datetime.now()
    
    # 대기 시간 이내 알림
    assert alert_manager._check_rate_limit('warning') == False
    
    # 대기 시간 이후 알림
    alert_manager.last_alert_time['warning'] = datetime.now() - timedelta(seconds=10)
    assert alert_manager._check_rate_limit('warning') == True

@pytest.mark.asyncio
async def test_alert_history(alert_manager):
    """알림 이력 테스트"""
    # 알림 기록
    metrics = {
        'latency': 1500,
        'error_rate': 0.08,
        'fill_rate': 0.92,
        'slippage': 0.002,
        'throughput': 25
    }
    
    # 알림 처리
    await alert_manager.process_metrics(metrics)
    await asyncio.sleep(0.1)
    
    # 이력 확인
    history = alert_manager.get_alert_history()
    assert len(history) > 0
    assert history[0]['level'] in ['warning', 'critical']
    
    # 레벨 필터링
    warning_history = alert_manager.get_alert_history(level='warning')
    critical_history = alert_manager.get_alert_history(level='critical')
    assert len(warning_history) + len(critical_history) == len(history)

@pytest.mark.asyncio
async def test_alert_stats(alert_manager):
    """알림 통계 테스트"""
    # 알림 기록
    metrics = {
        'latency': 2500,
        'error_rate': 0.15,
        'fill_rate': 0.4,
        'slippage': 0.005,
        'throughput': 10
    }
    
    # 알림 처리
    await alert_manager.process_metrics(metrics)
    await asyncio.sleep(0.1)
    
    # 통계 확인
    stats = alert_manager.get_alert_stats()
    assert stats['total_alerts'] > 0
    assert stats['daily_alerts'] > 0
    assert stats['critical_alerts'] > 0

@pytest.mark.asyncio
async def test_disabled_alerts(config):
    """알림 비활성화 테스트"""
    # 알림 비활성화 설정
    config['alerts']['enabled'] = False
    manager = AlertManager(config)
    await manager.initialize()
    
    # 알림 처리
    metrics = {
        'latency': 2500,
        'error_rate': 0.15,
        'fill_rate': 0.4,
        'slippage': 0.005,
        'throughput': 10
    }
    await manager.process_metrics(metrics)
    
    # 알림 이력 확인
    assert len(manager.alert_history) == 0
    
    await manager.close()

@pytest.mark.asyncio
async def test_error_handling(alert_manager):
    """에러 처리 테스트"""
    # 잘못된 메트릭
    invalid_metrics = {
        'latency': 'invalid',
        'error_rate': None,
        'fill_rate': 0.92,
        'slippage': 0.002,
        'throughput': 25
    }
    
    # 알림 처리
    await alert_manager.process_metrics(invalid_metrics)
    
    # 에러 처리 확인
    assert len(alert_manager.alert_history) == 0

@pytest.mark.asyncio
async def test_cleanup(alert_manager):
    """정리 테스트"""
    # 알림 기록
    metrics = {
        'latency': 1500,
        'error_rate': 0.08,
        'fill_rate': 0.92,
        'slippage': 0.002,
        'throughput': 25
    }
    await alert_manager.process_metrics(metrics)
    
    # 정리
    await alert_manager.close()
    
    # 정리 확인
    assert alert_manager.telegram is not None 