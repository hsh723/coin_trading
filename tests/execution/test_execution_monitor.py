"""
실행 모니터링 시스템 테스트

이 모듈은 실행 모니터링 시스템의 기능을 테스트합니다.
"""

import pytest
from datetime import datetime, timedelta
from src.execution.execution_monitor import ExecutionMonitor
from src.execution.quality_monitor import ExecutionQualityMonitor
from src.utils.logger import setup_logger

# 로거 설정
logger = setup_logger(name="test_execution_monitor")

@pytest.fixture
def monitor_config():
    """모니터링 설정"""
    return {
        'update_interval': 1.0,
        'window_size': 100,
        'latency_threshold': 1.0,
        'fill_rate_threshold': 0.95,
        'slippage_threshold': 0.001,
        'cost_threshold': 0.002,
        'latency_weight': 0.3,
        'fill_rate_weight': 0.3,
        'slippage_weight': 0.2,
        'cost_weight': 0.2
    }

@pytest.fixture
def execution_monitor(monitor_config):
    """실행 모니터 인스턴스"""
    return ExecutionMonitor(monitor_config)

@pytest.fixture
def quality_monitor(monitor_config):
    """품질 모니터 인스턴스"""
    return ExecutionQualityMonitor(monitor_config)

def test_execution_monitor_initialization(execution_monitor):
    """실행 모니터 초기화 테스트"""
    assert execution_monitor is not None
    assert execution_monitor.window_size == 100
    assert execution_monitor.update_interval == 1.0

def test_quality_monitor_initialization(quality_monitor):
    """품질 모니터 초기화 테스트"""
    assert quality_monitor is not None
    assert quality_monitor.window_size == 100
    assert quality_monitor.update_interval == 1.0
    assert quality_monitor.quality_threshold == 0.9

def test_update_metrics(execution_monitor):
    """메트릭 업데이트 테스트"""
    execution_data = {
        'latency': 0.5,
        'fill_rate': 0.98,
        'slippage': 0.0005,
        'cost': 0.001
    }
    
    execution_monitor.update_metrics(execution_data)
    metrics = execution_monitor.get_metrics()
    
    assert metrics['latency']['current'] == 0.5
    assert metrics['fill_rate']['current'] == 0.98
    assert metrics['slippage']['current'] == 0.0005
    assert metrics['cost']['current'] == 0.001

def test_update_quality(quality_monitor):
    """품질 업데이트 테스트"""
    execution_data = {
        'latency': 0.5,
        'fill_rate': 0.98,
        'slippage': 0.0005,
        'cost': 0.001
    }
    
    quality_monitor.update_quality(execution_data)
    metrics = quality_monitor.get_quality_metrics()
    
    assert metrics['quality']['current'] > 0
    assert metrics['quality']['current'] <= 1.0
    assert len(metrics['issues']['current']) == 0

def test_quality_issues(quality_monitor):
    """품질 이슈 감지 테스트"""
    execution_data = {
        'latency': 2.0,  # 임계값 초과
        'fill_rate': 0.7,  # 임계값 미달
        'slippage': 0.002,  # 임계값 초과
        'cost': 0.003  # 임계값 초과
    }
    
    quality_monitor.update_quality(execution_data)
    metrics = quality_monitor.get_quality_metrics()
    
    assert len(metrics['issues']['current']) > 0
    assert any(issue['type'] == 'latency' for issue in metrics['issues']['current'])
    assert any(issue['type'] == 'fill_rate' for issue in metrics['issues']['current'])
    assert any(issue['type'] == 'slippage' for issue in metrics['issues']['current'])
    assert any(issue['type'] == 'cost' for issue in metrics['issues']['current'])

def test_improvement_suggestions(quality_monitor):
    """개선 제안 테스트"""
    execution_data = {
        'latency': 2.0,
        'fill_rate': 0.7,
        'slippage': 0.002,
        'cost': 0.003
    }
    
    quality_monitor.update_quality(execution_data)
    suggestions = quality_monitor.get_improvement_suggestions()
    
    assert len(suggestions) > 0
    assert any("지연시간" in suggestion for suggestion in suggestions)
    assert any("체결률" in suggestion for suggestion in suggestions)
    assert any("슬리피지" in suggestion for suggestion in suggestions)
    assert any("실행 비용" in suggestion for suggestion in suggestions)

def test_health_status(execution_monitor, quality_monitor):
    """상태 확인 테스트"""
    # 정상 상태
    execution_data = {
        'latency': 0.5,
        'fill_rate': 0.98,
        'slippage': 0.0005,
        'cost': 0.001
    }
    
    execution_monitor.update_metrics(execution_data)
    quality_monitor.update_quality(execution_data)
    
    assert execution_monitor.is_healthy()
    assert quality_monitor.is_healthy()
    
    # 비정상 상태
    execution_data = {
        'latency': 2.0,
        'fill_rate': 0.7,
        'slippage': 0.002,
        'cost': 0.003
    }
    
    execution_monitor.update_metrics(execution_data)
    quality_monitor.update_quality(execution_data)
    
    assert not execution_monitor.is_healthy()
    assert not quality_monitor.is_healthy() 