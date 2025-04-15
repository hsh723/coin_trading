"""
오류 처리 모듈 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from src.execution.error_handler import ErrorHandler, ExecutionError

@pytest.fixture
def config():
    """테스트 설정"""
    return {
        'max_retries': 3,
        'retry_delay': 0.1,
        'max_history_size': 5,
        'notifications': {
            'enabled': False
        }
    }

@pytest.fixture
def error_handler(config):
    """오류 처리기 인스턴스"""
    return ErrorHandler(config)

@pytest.mark.asyncio
async def test_initialization(error_handler):
    """초기화 테스트"""
    await error_handler.initialize()
    assert error_handler.max_retries == 3
    assert error_handler.retry_delay == 0.1
    assert error_handler.max_history_size == 5
    assert len(error_handler.error_history) == 0
    await error_handler.close()

@pytest.mark.asyncio
async def test_handle_error(error_handler):
    """오류 처리 테스트"""
    await error_handler.initialize()
    
    # 기본 예외 처리
    error = Exception("테스트 오류")
    await error_handler.handle_error(error)
    
    assert len(error_handler.error_history) == 1
    error_info = error_handler.error_history[0]
    assert error_info['message'] == "테스트 오류"
    assert error_info['type'] == "Exception"
    
    await error_handler.close()

@pytest.mark.asyncio
async def test_handle_execution_error(error_handler):
    """실행 오류 처리 테스트"""
    await error_handler.initialize()
    
    # ExecutionError 처리
    error = ExecutionError(
        message="실행 오류",
        error_type="ORDER_FAILED",
        details={'order_id': '123'}
    )
    await error_handler.handle_error(error)
    
    assert len(error_handler.error_history) == 1
    error_info = error_handler.error_history[0]
    assert error_info['message'] == "실행 오류"
    assert error_info['error_type'] == "ORDER_FAILED"
    assert error_info['details']['order_id'] == '123'
    
    await error_handler.close()

@pytest.mark.asyncio
async def test_error_callback(error_handler):
    """오류 콜백 테스트"""
    await error_handler.initialize()
    
    callback_called = False
    callback_error = None
    
    async def error_callback(error_info: Dict[str, Any]):
        nonlocal callback_called, callback_error
        callback_called = True
        callback_error = error_info
    
    # 콜백 등록
    error_handler.register_error_callback("TEST_ERROR", error_callback)
    
    # 오류 발생
    error = ExecutionError(
        message="테스트 오류",
        error_type="TEST_ERROR"
    )
    await error_handler.handle_error(error)
    
    assert callback_called
    assert callback_error['error_type'] == "TEST_ERROR"
    
    await error_handler.close()

@pytest.mark.asyncio
async def test_error_history_limit(error_handler):
    """오류 이력 제한 테스트"""
    await error_handler.initialize()
    
    # 최대 크기보다 많은 오류 발생
    for i in range(10):
        error = Exception(f"오류 {i}")
        await error_handler.handle_error(error)
    
    assert len(error_handler.error_history) == 5  # max_history_size
    assert error_handler.error_history[-1]['message'] == "오류 9"
    
    await error_handler.close()

@pytest.mark.asyncio
async def test_get_error_history(error_handler):
    """오류 이력 조회 테스트"""
    await error_handler.initialize()
    
    # 테스트 데이터 생성
    now = datetime.now()
    
    # 과거 오류
    error1 = ExecutionError(
        message="과거 오류",
        error_type="TYPE_A"
    )
    error1.timestamp = now - timedelta(hours=2)
    await error_handler.handle_error(error1)
    
    # 현재 오류
    error2 = ExecutionError(
        message="현재 오류",
        error_type="TYPE_B"
    )
    await error_handler.handle_error(error2)
    
    # 시간 범위로 필터링
    history = error_handler.get_error_history(
        start_time=now - timedelta(hours=1)
    )
    assert len(history) == 1
    assert history[0]['message'] == "현재 오류"
    
    # 타입으로 필터링
    history = error_handler.get_error_history(error_type="TYPE_A")
    assert len(history) == 1
    assert history[0]['message'] == "과거 오류"
    
    await error_handler.close()

@pytest.mark.asyncio
async def test_get_error_stats(error_handler):
    """오류 통계 조회 테스트"""
    await error_handler.initialize()
    
    # 테스트 데이터 생성
    errors = [
        ExecutionError("오류 1", "TYPE_A"),
        ExecutionError("오류 2", "TYPE_A"),
        ExecutionError("오류 3", "TYPE_B")
    ]
    
    for error in errors:
        await error_handler.handle_error(error)
    
    # 통계 조회
    stats = error_handler.get_error_stats()
    
    assert stats['total_errors'] == 3
    assert stats['error_counts']['TYPE_A'] == 2
    assert stats['error_counts']['TYPE_B'] == 1
    
    await error_handler.close()

@pytest.mark.asyncio
async def test_clear_error_history(error_handler):
    """오류 이력 초기화 테스트"""
    await error_handler.initialize()
    
    # 오류 발생
    error = Exception("테스트 오류")
    await error_handler.handle_error(error)
    
    assert len(error_handler.error_history) == 1
    
    # 이력 초기화
    error_handler.clear_error_history()
    assert len(error_handler.error_history) == 0
    
    await error_handler.close()

@pytest.mark.asyncio
async def test_error_context(error_handler):
    """오류 컨텍스트 테스트"""
    await error_handler.initialize()
    
    # 컨텍스트와 함께 오류 발생
    context = {
        'order_id': '123',
        'symbol': 'BTC/USDT',
        'side': 'BUY'
    }
    
    error = Exception("주문 실패")
    await error_handler.handle_error(error, context)
    
    assert len(error_handler.error_history) == 1
    error_info = error_handler.error_history[0]
    assert error_info['context'] == context
    
    await error_handler.close() 