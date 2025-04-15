"""
실시간 모니터 테스트
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import aiofiles
import aiofiles.os
from src.execution.real_time_monitor import RealTimeMonitor

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
        },
        'real_time': {
            'update_interval': 1,  # 1초
            'window_size': 60,  # 1분
            'alert_thresholds': {
                'spread': 0.001,  # 0.1%
                'volatility': 0.02,  # 2%
                'volume_imbalance': 0.7,  # 70%
                'market_impact': 0.001  # 0.1%
            }
        }
    }

@pytest.fixture
async def monitor(config):
    """모니터 인스턴스"""
    monitor = RealTimeMonitor(config)
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
    monitor = RealTimeMonitor(config)
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
async def test_real_time_metrics_collection(monitor):
    """실시간 메트릭 수집 테스트"""
    # 실행 데이터 기록
    execution_data = {
        'spread': 0.0005,
        'price': 50000.0,
        'volume': 1.0,
        'side': 'buy',
        'market_impact': 0.0002,
        'depth': 100.0
    }
    await monitor.logger.log_execution(execution_data)
    
    # 메트릭 수집
    await monitor._collect_real_time_metrics()
    
    # 메트릭 확인
    assert len(monitor.metrics['spread']) > 0
    assert len(monitor.metrics['volatility']) > 0
    assert len(monitor.metrics['volume_imbalance']) > 0
    assert len(monitor.metrics['market_impact']) > 0
    assert len(monitor.metrics['order_flow']) > 0
    assert len(monitor.metrics['liquidity_score']) > 0

@pytest.mark.asyncio
async def test_market_state_analysis(monitor):
    """시장 상태 분석 테스트"""
    # 정상 데이터 기록
    for _ in range(60):
        await monitor.logger.log_execution({
            'spread': 0.0005,
            'price': 50000.0,
            'volume': 1.0,
            'side': 'buy',
            'market_impact': 0.0002,
            'depth': 100.0
        })
    
    # 이상 데이터 기록
    await monitor.logger.log_execution({
        'spread': 0.002,  # 임계값 초과
        'price': 50000.0,
        'volume': 1.0,
        'side': 'buy',
        'market_impact': 0.002,  # 임계값 초과
        'depth': 100.0
    })
    
    # 메트릭 수집 및 상태 분석
    await monitor._collect_real_time_metrics()
    await monitor._analyze_market_state()
    
    # 시장 상태 확인
    market_state = monitor.get_market_state()
    assert 'is_volatile' in market_state
    assert 'is_liquid' in market_state
    assert 'is_stable' in market_state
    assert 'regime' in market_state

@pytest.mark.asyncio
async def test_execution_quality_monitoring(monitor):
    """실행 품질 모니터링 테스트"""
    # 실행 데이터 기록
    for i in range(10):
        await monitor.logger.log_execution({
            'fill_rate': 0.98,
            'price_improvement': 0.0001,
            'timing_score': 0.8,
            'cost_score': 0.9,
            'spread': 0.0005,
            'price': 50000.0,
            'volume': 1.0,
            'side': 'buy',
            'market_impact': 0.0002,
            'depth': 100.0
        })
    
    # 실행 품질 모니터링
    await monitor._monitor_execution_quality()
    
    # 품질 로그 확인
    logs = await monitor.get_execution_quality()
    assert len(logs) > 0
    assert 'score' in logs.columns

@pytest.mark.asyncio
async def test_liquidity_score_calculation(monitor):
    """유동성 점수 계산 테스트"""
    # 테스트 데이터
    logs = pd.DataFrame({
        'spread': [0.0005] * 10,
        'depth': [100.0] * 10,
        'volume': [1.0] * 10
    })
    
    # 유동성 점수 계산
    score = monitor._calculate_liquidity_score(logs)
    
    # 점수 확인
    assert 0 <= score <= 1

@pytest.mark.asyncio
async def test_market_regime_determination(monitor):
    """시장 레짐 판단 테스트"""
    # 시장 상태 설정
    monitor.market_state['is_volatile'] = True
    monitor.market_state['is_liquid'] = True
    
    # 레짐 판단
    regime = monitor._determine_market_regime()
    
    # 레짐 확인
    assert regime in ['volatile_liquid', 'volatile_illiquid', 'stable_liquid', 'stable_illiquid', 'normal']

@pytest.mark.asyncio
async def test_real_time_metrics_history(monitor):
    """실시간 메트릭 이력 테스트"""
    # 실행 데이터 기록
    start_time = datetime.now()
    for i in range(5):
        await monitor.logger.log_execution({
            'spread': 0.0005,
            'price': 50000.0,
            'volume': 1.0,
            'side': 'buy',
            'market_impact': 0.0002,
            'depth': 100.0
        })
        await asyncio.sleep(0.1)
    end_time = datetime.now()
    
    # 메트릭 수집
    await monitor._collect_real_time_metrics()
    
    # 메트릭 이력 조회
    history = await monitor.get_real_time_metrics(
        start_time=start_time,
        end_time=end_time
    )
    assert len(history) > 0

@pytest.mark.asyncio
async def test_monitoring_loop(monitor):
    """모니터링 루프 테스트"""
    # 실행 데이터 기록
    await monitor.logger.log_execution({
        'spread': 0.0005,
        'price': 50000.0,
        'volume': 1.0,
        'side': 'buy',
        'market_impact': 0.0002,
        'depth': 100.0
    })
    
    # 모니터링 루프 실행 (1초)
    await asyncio.sleep(1.1)
    
    # 메트릭 확인
    market_state = monitor.get_market_state()
    assert market_state['regime'] != ''

@pytest.mark.asyncio
async def test_cleanup(monitor):
    """정리 테스트"""
    # 실행 데이터 기록
    await monitor.logger.log_execution({
        'spread': 0.0005,
        'price': 50000.0,
        'volume': 1.0,
        'side': 'buy',
        'market_impact': 0.0002,
        'depth': 100.0
    })
    
    # 모니터 종료
    await monitor.close()
    
    # 로거 핸들러 정리 확인
    assert len(monitor.logger.logger.handlers) == 0 