"""
실행 시스템 로거 테스트
"""

import pytest
import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import aiofiles
import aiofiles.os
from src.execution.logger import ExecutionLogger

@pytest.fixture
def config():
    """테스트 설정"""
    return {
        'logging': {
            'level': 'INFO',
            'log_dir': 'logs/test',
            'max_log_size': 1024,  # 1KB
            'backup_count': 3
        }
    }

@pytest.fixture
async def logger(config):
    """로거 인스턴스"""
    logger = ExecutionLogger(config)
    await logger.initialize()
    yield logger
    await logger.close()
    
    # 테스트 로그 파일 정리
    log_dir = Path(config['logging']['log_dir'])
    if await aiofiles.os.path.exists(log_dir):
        for log_file in log_dir.glob('*.log*'):
            await aiofiles.os.remove(log_file)
        await aiofiles.os.rmdir(log_dir)

@pytest.mark.asyncio
async def test_initialization(config):
    """초기화 테스트"""
    logger = ExecutionLogger(config)
    await logger.initialize()
    
    # 로그 디렉토리 생성 확인
    log_dir = Path(config['logging']['log_dir'])
    assert await aiofiles.os.path.exists(log_dir)
    
    # 로그 파일 생성 확인
    assert await aiofiles.os.path.exists(log_dir / 'execution.log')
    assert await aiofiles.os.path.exists(log_dir / 'performance.log')
    assert await aiofiles.os.path.exists(log_dir / 'error.log')
    
    await logger.close()

@pytest.mark.asyncio
async def test_log_execution(logger):
    """실행 로그 기록 테스트"""
    # 실행 데이터
    execution_data = {
        'order_id': '123',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'quantity': 1.0,
        'price': 50000.0
    }
    
    # 로그 기록
    await logger.log_execution(execution_data)
    
    # 로그 파일 확인
    log_file = Path(logger.execution_log)
    assert await aiofiles.os.path.exists(log_file)
    
    # 로그 내용 확인
    async with aiofiles.open(log_file, 'r') as f:
        log_entry = json.loads(await f.readline())
        
    assert log_entry['order_id'] == '123'
    assert log_entry['symbol'] == 'BTC/USDT'
    assert 'timestamp' in log_entry

@pytest.mark.asyncio
async def test_log_performance(logger):
    """성능 로그 기록 테스트"""
    # 성능 데이터
    performance_data = {
        'latency': 50.0,
        'success_rate': 0.95,
        'fill_rate': 0.98
    }
    
    # 로그 기록
    await logger.log_performance(performance_data)
    
    # 로그 파일 확인
    log_file = Path(logger.performance_log)
    assert await aiofiles.os.path.exists(log_file)
    
    # 로그 내용 확인
    async with aiofiles.open(log_file, 'r') as f:
        log_entry = json.loads(await f.readline())
        
    assert log_entry['latency'] == 50.0
    assert log_entry['success_rate'] == 0.95
    assert 'timestamp' in log_entry

@pytest.mark.asyncio
async def test_log_error(logger):
    """오류 로그 기록 테스트"""
    # 오류 데이터
    error_data = {
        'code': 'E001',
        'message': '주문 실행 실패',
        'details': {
            'order_id': '123',
            'reason': 'insufficient_funds'
        }
    }
    
    # 로그 기록
    await logger.log_error(error_data)
    
    # 로그 파일 확인
    log_file = Path(logger.error_log)
    assert await aiofiles.os.path.exists(log_file)
    
    # 로그 내용 확인
    async with aiofiles.open(log_file, 'r') as f:
        log_entry = json.loads(await f.readline())
        
    assert log_entry['code'] == 'E001'
    assert log_entry['message'] == '주문 실행 실패'
    assert 'timestamp' in log_entry

@pytest.mark.asyncio
async def test_log_rotation(logger):
    """로그 순환 테스트"""
    # 큰 로그 데이터 생성
    large_data = {
        'data': 'x' * 1000  # 1KB 이상의 데이터
    }
    
    # 여러 번 로그 기록
    for _ in range(5):
        await logger.log_execution(large_data)
    
    # 백업 파일 확인
    log_file = Path(logger.execution_log)
    backup1 = log_file.with_suffix('.log.1')
    backup2 = log_file.with_suffix('.log.2')
    
    assert await aiofiles.os.path.exists(backup1)
    assert await aiofiles.os.path.exists(backup2)

@pytest.mark.asyncio
async def test_get_execution_logs(logger):
    """실행 로그 조회 테스트"""
    # 테스트 데이터
    now = datetime.now()
    
    execution_data1 = {
        'order_id': '123',
        'timestamp': (now - timedelta(hours=2)).isoformat()
    }
    execution_data2 = {
        'order_id': '456',
        'timestamp': now.isoformat()
    }
    
    # 로그 기록
    await logger.log_execution(execution_data1)
    await logger.log_execution(execution_data2)
    
    # 전체 로그 조회
    logs = await logger.get_execution_logs()
    assert len(logs) == 2
    
    # 시간 범위 지정 조회
    start_time = now - timedelta(hours=1)
    logs = await logger.get_execution_logs(start_time=start_time)
    assert len(logs) == 1
    assert logs.iloc[0]['order_id'] == '456'

@pytest.mark.asyncio
async def test_get_performance_logs(logger):
    """성능 로그 조회 테스트"""
    # 테스트 데이터
    now = datetime.now()
    
    performance_data1 = {
        'latency': 50.0,
        'timestamp': (now - timedelta(hours=2)).isoformat()
    }
    performance_data2 = {
        'latency': 60.0,
        'timestamp': now.isoformat()
    }
    
    # 로그 기록
    await logger.log_performance(performance_data1)
    await logger.log_performance(performance_data2)
    
    # 전체 로그 조회
    logs = await logger.get_performance_logs()
    assert len(logs) == 2
    
    # 시간 범위 지정 조회
    end_time = now - timedelta(hours=1)
    logs = await logger.get_performance_logs(end_time=end_time)
    assert len(logs) == 1
    assert logs.iloc[0]['latency'] == 50.0

@pytest.mark.asyncio
async def test_get_error_logs(logger):
    """오류 로그 조회 테스트"""
    # 테스트 데이터
    now = datetime.now()
    
    error_data1 = {
        'code': 'E001',
        'timestamp': (now - timedelta(hours=2)).isoformat()
    }
    error_data2 = {
        'code': 'E002',
        'timestamp': now.isoformat()
    }
    
    # 로그 기록
    await logger.log_error(error_data1)
    await logger.log_error(error_data2)
    
    # 전체 로그 조회
    logs = await logger.get_error_logs()
    assert len(logs) == 2
    
    # 시간 범위 지정 조회
    start_time = now - timedelta(hours=1)
    end_time = now + timedelta(hours=1)
    logs = await logger.get_error_logs(
        start_time=start_time,
        end_time=end_time
    )
    assert len(logs) == 1
    assert logs.iloc[0]['code'] == 'E002'

@pytest.mark.asyncio
async def test_cleanup(logger):
    """정리 테스트"""
    # 로그 기록
    await logger.log_execution({'test': 'data'})
    await logger.log_performance({'test': 'data'})
    await logger.log_error({'test': 'data'})
    
    # 로거 종료
    await logger.close()
    
    # 핸들러 정리 확인
    assert len(logger.logger.handlers) == 0 