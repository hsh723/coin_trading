"""
실행 시스템 모니터 테스트
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import aiofiles
import aiofiles.os
from src.execution.monitor import ExecutionMonitor

@pytest.fixture
def config():
    """테스트 설정"""
    return {
        'logging': {
            'level': 'INFO',
            'log_dir': 'logs/test',
            'max_log_size': 1024,  # 1KB
            'backup_count': 3
        },
        'monitoring': {
            'interval': 1,  # 1초
            'alert_thresholds': {
                'latency': 1000,  # ms
                'error_rate': 0.05,  # 5%
                'fill_rate': 0.95,  # 95%
                'slippage': 0.001  # 0.1%
            }
        }
    }

@pytest.fixture
async def monitor(config):
    """모니터 인스턴스"""
    monitor = ExecutionMonitor(config)
    await monitor.initialize()
    yield monitor
    await monitor.close()
    
    # 테스트 로그 파일 정리
    log_dir = Path(config['logging']['log_dir'])
    if await aiofiles.os.path.exists(log_dir):
        for log_file in log_dir.glob('*.log*'):
            await aiofiles.os.remove(log_file)
        await aiofiles.os.rmdir(log_dir)

@pytest.mark.asyncio
async def test_initialization(config):
    """초기화 테스트"""
    monitor = ExecutionMonitor(config)
    await monitor.initialize()
    
    # 로그 디렉토리 생성 확인
    log_dir = Path(config['logging']['log_dir'])
    assert await aiofiles.os.path.exists(log_dir)
    
    # 로그 파일 생성 확인
    assert await aiofiles.os.path.exists(log_dir / 'execution.log')
    assert await aiofiles.os.path.exists(log_dir / 'performance.log')
    assert await aiofiles.os.path.exists(log_dir / 'error.log')
    
    await monitor.close()

@pytest.mark.asyncio
async def test_metrics_collection(monitor):
    """메트릭 수집 테스트"""
    # 실행 데이터 기록
    execution_data = {
        'latency': 100.0,
        'fill_rate': 0.98,
        'slippage': 0.0005,
        'volume': 1.0
    }
    await monitor.logger.log_execution(execution_data)
    
    # 메트릭 수집
    await monitor._collect_metrics()
    
    # 메트릭 확인
    metrics = monitor.get_metrics_summary()
    assert 'latency' in metrics
    assert 'fill_rate' in metrics
    assert 'slippage' in metrics
    assert 'volume' in metrics

@pytest.mark.asyncio
async def test_anomaly_detection(monitor):
    """이상 탐지 테스트"""
    # 정상 데이터 기록
    for _ in range(60):
        await monitor.logger.log_execution({
            'latency': 100.0,
            'fill_rate': 0.98,
            'slippage': 0.0005,
            'volume': 1.0
        })
    
    # 이상 데이터 기록
    await monitor.logger.log_execution({
        'latency': 2000.0,  # 임계값 초과
        'fill_rate': 0.98,
        'slippage': 0.0005,
        'volume': 1.0
    })
    
    # 메트릭 수집 및 이상 탐지
    await monitor._collect_metrics()
    await monitor._detect_anomalies()
    
    # 이상 탐지 로그 확인
    anomalies = await monitor.get_anomalies()
    assert len(anomalies) > 0
    assert anomalies.iloc[0]['code'] == 'ANOMALY'

@pytest.mark.asyncio
async def test_performance_analysis(monitor):
    """성능 분석 테스트"""
    # 실행 데이터 기록
    for i in range(10):
        await monitor.logger.log_execution({
            'latency': 100.0 + i,
            'fill_rate': 0.98,
            'slippage': 0.0005,
            'volume': 1.0
        })
    
    # 메트릭 수집 및 성능 분석
    await monitor._collect_metrics()
    await monitor._analyze_performance()
    
    # 성능 로그 확인
    logs = await monitor.logger.get_performance_logs()
    assert len(logs) > 0
    assert 'metrics' in logs.iloc[-1]

@pytest.mark.asyncio
async def test_metrics_history(monitor):
    """메트릭 이력 테스트"""
    # 실행 데이터 기록
    start_time = datetime.now()
    for i in range(5):
        await monitor.logger.log_execution({
            'latency': 100.0 + i,
            'fill_rate': 0.98,
            'slippage': 0.0005,
            'volume': 1.0
        })
        await asyncio.sleep(0.1)
    end_time = datetime.now()
    
    # 메트릭 수집
    await monitor._collect_metrics()
    
    # 메트릭 이력 조회
    history = await monitor.get_metrics_history(
        start_time=start_time,
        end_time=end_time
    )
    assert len(history) > 0
    assert 'latency' in history.columns

@pytest.mark.asyncio
async def test_error_handling(monitor):
    """오류 처리 테스트"""
    # 오류 데이터 기록
    error_data = {
        'code': 'TEST_ERROR',
        'message': '테스트 오류',
        'details': {'reason': 'test'}
    }
    await monitor.logger.log_error(error_data)
    
    # 실행 데이터 기록 (오류 발생)
    await monitor.logger.log_execution({
        'latency': 100.0,
        'fill_rate': 0.98,
        'slippage': 0.0005,
        'volume': 1.0,
        'error': True
    })
    
    # 메트릭 수집
    await monitor._collect_metrics()
    
    # 오류율 확인
    metrics = monitor.get_metrics_summary()
    assert metrics['error_rate']['current'] > 0

@pytest.mark.asyncio
async def test_monitoring_loop(monitor):
    """모니터링 루프 테스트"""
    # 실행 데이터 기록
    await monitor.logger.log_execution({
        'latency': 100.0,
        'fill_rate': 0.98,
        'slippage': 0.0005,
        'volume': 1.0
    })
    
    # 모니터링 루프 실행 (1초)
    await asyncio.sleep(1.1)
    
    # 메트릭 확인
    metrics = monitor.get_metrics_summary()
    assert len(metrics) > 0

@pytest.mark.asyncio
async def test_cleanup(monitor):
    """정리 테스트"""
    # 실행 데이터 기록
    await monitor.logger.log_execution({
        'latency': 100.0,
        'fill_rate': 0.98,
        'slippage': 0.0005,
        'volume': 1.0
    })
    
    # 모니터 종료
    await monitor.close()
    
    # 로거 핸들러 정리 확인
    assert len(monitor.logger.logger.handlers) == 0 