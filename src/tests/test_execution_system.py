"""
실행 시스템 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from typing import Dict

from src.execution.manager import ExecutionManager
from src.execution.algorithms.twap_executor import TWAPExecutor
from src.execution.algorithms.vwap_executor import VWAPExecutor
from src.execution.algorithms.is_executor import ISExecutor
from src.execution.algorithms.pov_executor import POVExecutor
from src.execution.algorithms.adaptive_executor import AdaptiveExecutor

@pytest.fixture
async def execution_manager():
    """실행 관리자 픽스처"""
    config = {
        'default_strategy': 'adaptive',
        'risk_limit': 0.02,
        'max_slippage': 0.003,
        'execution_timeout': 300,
        'retry_limit': 3,
        'concurrent_limit': 5
    }
    manager = ExecutionManager(config)
    yield manager
    
@pytest.fixture
def sample_order():
    """샘플 주문 픽스처"""
    return {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'amount': 1.0,
        'price': 50000.0,
        'leverage': 1.0,
        'time_window': 3600,
        'volume_profile': [0.1] * 10,
        'urgency': 0.5,
        'participation_rate': 0.1
    }
    
@pytest.fixture
def sample_market_data():
    """샘플 시장 데이터 픽스처"""
    return {
        'price': 50000.0,
        'spread': 0.001,
        'depth': 100.0,
        'volume': 1000.0,
        'volatility': 0.02,
        'buy_volume': 600.0,
        'sell_volume': 400.0,
        'daily_volume': 10000.0,
        'historical_prices': [
            50000.0 * (1 + 0.0001 * i) for i in range(100)
        ]
    }
    
@pytest.mark.asyncio
async def test_execution_manager_initialization(execution_manager):
    """실행 관리자 초기화 테스트"""
    assert execution_manager.config['default_strategy'] == 'adaptive'
    assert len(execution_manager.executors) == 5
    assert isinstance(execution_manager.executors['twap'], TWAPExecutor)
    assert isinstance(execution_manager.executors['vwap'], VWAPExecutor)
    assert isinstance(execution_manager.executors['is'], ISExecutor)
    assert isinstance(execution_manager.executors['pov'], POVExecutor)
    assert isinstance(execution_manager.executors['adaptive'], AdaptiveExecutor)
    
@pytest.mark.asyncio
async def test_order_validation(execution_manager, sample_order):
    """주문 검증 테스트"""
    # 유효한 주문
    await execution_manager._validate_execution(sample_order, 'twap')
    
    # 심볼 누락
    invalid_order = sample_order.copy()
    del invalid_order['symbol']
    with pytest.raises(ValueError, match="거래 심볼 누락"):
        await execution_manager._validate_execution(invalid_order, 'twap')
        
    # 수량 누락
    invalid_order = sample_order.copy()
    del invalid_order['amount']
    with pytest.raises(ValueError, match="거래 수량 누락"):
        await execution_manager._validate_execution(invalid_order, 'twap')
        
@pytest.mark.asyncio
async def test_risk_exposure_calculation(execution_manager, sample_order, sample_market_data):
    """위험 노출도 계산 테스트"""
    # 기본 위험 노출도
    sample_order['market_data'] = sample_market_data
    risk = await execution_manager._calculate_risk_exposure(sample_order)
    assert 0 <= risk <= 1
    
    # 레버리지 증가
    high_leverage_order = sample_order.copy()
    high_leverage_order['leverage'] = 5.0
    high_risk = await execution_manager._calculate_risk_exposure(high_leverage_order)
    assert high_risk > risk
    
    # 변동성 증가
    high_vol_market = sample_market_data.copy()
    high_vol_market['volatility'] = 0.05
    high_vol_order = sample_order.copy()
    high_vol_order['market_data'] = high_vol_market
    high_vol_risk = await execution_manager._calculate_risk_exposure(high_vol_order)
    assert high_vol_risk > risk
    
@pytest.mark.asyncio
async def test_strategy_constraints(execution_manager, sample_order):
    """전략별 제약 조건 테스트"""
    # TWAP 전략
    await execution_manager._validate_strategy_constraints(sample_order, 'twap')
    invalid_order = sample_order.copy()
    del invalid_order['time_window']
    with pytest.raises(ValueError, match="TWAP 전략에 time_window 필요"):
        await execution_manager._validate_strategy_constraints(invalid_order, 'twap')
        
    # VWAP 전략
    await execution_manager._validate_strategy_constraints(sample_order, 'vwap')
    invalid_order = sample_order.copy()
    del invalid_order['volume_profile']
    with pytest.raises(ValueError, match="VWAP 전략에 volume_profile 필요"):
        await execution_manager._validate_strategy_constraints(invalid_order, 'vwap')
        
@pytest.mark.asyncio
async def test_performance_metrics(execution_manager):
    """성과 지표 계산 테스트"""
    execution = {
        'start_time': datetime.now() - timedelta(minutes=5),
        'order': {
            'amount': 1.0,
            'price': 50000.0
        }
    }
    
    results = {
        'implementation_shortfall': 0.001,
        'average_price': 49900.0,
        'total_executed': 0.99
    }
    
    metrics = execution_manager._calculate_performance_metrics(execution, results)
    
    assert 'execution_time' in metrics
    assert 'implementation_shortfall' in metrics
    assert 'price_improvement' in metrics
    assert 'completion_rate' in metrics
    assert 'slippage' in metrics
    
    assert metrics['implementation_shortfall'] == 0.001
    assert metrics['completion_rate'] == 0.99
    assert abs(metrics['price_improvement']) <= 0.01
    
@pytest.mark.asyncio
async def test_anomaly_detection(execution_manager):
    """이상 감지 테스트"""
    # 정상 지표
    normal_metrics = {
        'execution_time': 100,
        'slippage': 0.001,
        'completion_rate': 1.0
    }
    anomalies = execution_manager._detect_anomalies(normal_metrics)
    assert not anomalies
    
    # 실행 시간 초과
    slow_metrics = normal_metrics.copy()
    slow_metrics['execution_time'] = 1000
    anomalies = execution_manager._detect_anomalies(slow_metrics)
    assert 'excessive_execution_time' in anomalies
    
    # 높은 슬리피지
    high_slip_metrics = normal_metrics.copy()
    high_slip_metrics['slippage'] = 0.01
    anomalies = execution_manager._detect_anomalies(high_slip_metrics)
    assert 'excessive_slippage' in anomalies
    
    # 낮은 완료율
    incomplete_metrics = normal_metrics.copy()
    incomplete_metrics['completion_rate'] = 0.95
    anomalies = execution_manager._detect_anomalies(incomplete_metrics)
    assert 'incomplete_execution' in anomalies
    
@pytest.mark.asyncio
async def test_execution_stats(execution_manager, sample_order, sample_market_data):
    """실행 통계 테스트"""
    # 실행 이력 생성
    for i in range(5):
        execution_id = f"test_exec_{i}"
        execution_manager.execution_history[execution_id] = {
            'execution': {
                'order': sample_order,
                'market_data': sample_market_data,
                'start_time': datetime.now() - timedelta(minutes=5)
            },
            'results': {
                'implementation_shortfall': 0.001,
                'average_price': 49900.0,
                'total_executed': 0.99
            },
            'metrics': {
                'execution_time': 100,
                'slippage': 0.001,
                'completion_rate': 0.99
            },
            'anomalies': [] if i < 4 else ['excessive_slippage'],
            'completion_time': datetime.now()
        }
        
    stats = execution_manager.get_execution_stats()
    
    assert stats['total_executions'] == 5
    assert stats['success_rate'] == 0.8
    assert 'average_metrics' in stats
    assert 'anomaly_counts' in stats
    assert stats['anomaly_counts'].get('excessive_slippage', 0) == 1
    
@pytest.mark.asyncio
async def test_concurrent_execution_limit(execution_manager, sample_order, sample_market_data):
    """동시 실행 제한 테스트"""
    # 최대 동시 실행 수만큼 실행 생성
    for i in range(execution_manager.config['concurrent_limit']):
        execution_id = f"test_exec_{i}"
        execution_manager.active_executions[execution_id] = {
            'order': sample_order,
            'strategy': 'twap',
            'status': 'running',
            'start_time': datetime.now(),
            'market_data': sample_market_data
        }
        
    # 추가 실행 시도
    with pytest.raises(RuntimeError, match="동시 실행 한도 초과"):
        await execution_manager.execute_order(sample_order, 'twap', sample_market_data)
        
@pytest.mark.asyncio
async def test_execution_retry(execution_manager, sample_order, sample_market_data):
    """실행 재시도 테스트"""
    class FailingExecutor:
        fail_count = 0
        
        async def execute_order(self, order, market_data):
            self.fail_count += 1
            if self.fail_count < 3:
                raise Exception("일시적 오류")
            return {'success': True}
            
    # 실패하는 실행기로 교체
    execution_manager.executors['test'] = FailingExecutor()
    
    # 재시도 후 성공
    result = await execution_manager._execute_with_retry(
        execution_manager.executors['test'],
        'test_exec',
        sample_order,
        sample_market_data
    )
    
    assert result['success']
    assert execution_manager.executors['test'].fail_count == 3
    
    # 최대 재시도 횟수 초과
    execution_manager.executors['test'] = FailingExecutor()
    execution_manager.config['retry_limit'] = 2
    
    with pytest.raises(RuntimeError, match="최대 재시도 횟수 초과"):
        await execution_manager._execute_with_retry(
            execution_manager.executors['test'],
            'test_exec',
            sample_order,
            sample_market_data
        ) 