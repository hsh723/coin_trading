"""
실행 전략 최적화기 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from src.execution.strategy_optimizer import ExecutionStrategyOptimizer

@pytest.fixture
def config():
    """테스트 설정"""
    return {
        'strategies': ['twap', 'vwap', 'market', 'limit', 'iceberg', 'adaptive'],
        'optimization_interval': 1,  # 1시간마다 최적화 (테스트용)
        'min_samples': 3,  # 테스트를 위한 최소 샘플 수
        'weight_decay': 0.95,
        'exploration_rate': 0.1,
        'load_history': False,
        'save_history': False
    }

@pytest.fixture
async def optimizer(config):
    """최적화기 인스턴스"""
    opt = ExecutionStrategyOptimizer(config)
    await opt.initialize()
    yield opt
    await opt.close()

@pytest.mark.asyncio
async def test_initialization(config):
    """초기화 테스트"""
    optimizer = ExecutionStrategyOptimizer(config)
    await optimizer.initialize()
    
    # 설정 확인
    assert optimizer.strategies == config['strategies']
    assert optimizer.optimization_interval == config['optimization_interval']
    assert optimizer.min_samples == config['min_samples']
    assert optimizer.weight_decay == config['weight_decay']
    assert optimizer.exploration_rate == config['exploration_rate']
    
    # 가중치 초기화 확인
    assert len(optimizer.strategy_weights) > 0
    assert sum(optimizer.strategy_weights['normal'].values()) == pytest.approx(1.0)
    
    # 작업 시작 확인
    assert optimizer.is_running == True
    assert optimizer.optimization_task is not None
    
    await optimizer.close()

@pytest.mark.asyncio
async def test_add_execution_result(optimizer):
    """실행 결과 추가 테스트"""
    # 초기 상태 확인
    market_state = 'normal'
    strategy = 'twap'
    assert len(optimizer.performance_history[market_state][strategy]) == 0
    
    # 결과 추가
    score1 = 0.8
    optimizer.add_execution_result(strategy, market_state, score1)
    assert len(optimizer.performance_history[market_state][strategy]) == 1
    assert optimizer.performance_history[market_state][strategy][0] == score1
    
    # 결과 추가 (여러 개)
    score2 = 0.7
    optimizer.add_execution_result(strategy, market_state, score2)
    assert len(optimizer.performance_history[market_state][strategy]) == 2
    assert optimizer.performance_history[market_state][strategy][1] == score2
    
    # 잘못된 전략
    optimizer.add_execution_result('unknown_strategy', market_state, 0.5)
    assert 'unknown_strategy' not in optimizer.performance_history[market_state]
    
    # 잘못된 시장 상태
    optimizer.add_execution_result(strategy, 'unknown_state', 0.5)
    assert 'unknown_state' not in optimizer.performance_history

@pytest.mark.asyncio
async def test_get_optimal_strategy(optimizer):
    """최적 전략 선택 테스트"""
    # 기본 시장 상태
    market_state = 'normal'
    strategy = optimizer.get_optimal_strategy(market_state)
    assert strategy in optimizer.strategies
    
    # 알 수 없는 시장 상태
    strategy = optimizer.get_optimal_strategy('unknown_state')
    assert strategy in optimizer.strategies  # 기본값 반환
    
    # 주문 상세 정보 포함
    order_details = {
        'quantity': 10.0,
        'market_volume': 100.0,  # 10%
        'urgent': True,
        'spread': 0.003  # 0.3%
    }
    strategy = optimizer.get_optimal_strategy(market_state, order_details)
    assert strategy in optimizer.strategies

@pytest.mark.asyncio
async def test_adjust_weights_for_order(optimizer):
    """주문별 가중치 조정 테스트"""
    market_state = 'normal'
    original_weights = optimizer.strategy_weights[market_state].copy()
    
    # 대형 주문 (시장 볼륨의 10%)
    order_details = {
        'quantity': 10.0,
        'market_volume': 100.0
    }
    
    adjusted_weights = optimizer._adjust_weights_for_order(market_state, order_details)
    
    # 아이스버그 및 TWAP 전략 가중치가 증가해야 함
    assert adjusted_weights['iceberg'] > original_weights['iceberg']
    assert adjusted_weights['twap'] > original_weights['twap']
    
    # 가중치 합이 1이어야 함
    assert sum(adjusted_weights.values()) == pytest.approx(1.0)
    
    # 긴급 주문
    order_details = {
        'urgent': True
    }
    
    adjusted_weights = optimizer._adjust_weights_for_order(market_state, order_details)
    
    # 시장가 및 적응형 전략 가중치가 증가해야 함
    assert adjusted_weights['market'] > original_weights['market']
    assert adjusted_weights['adaptive'] > original_weights['adaptive']
    
    # 넓은 스프레드
    order_details = {
        'spread': 0.003  # 0.3%
    }
    
    adjusted_weights = optimizer._adjust_weights_for_order(market_state, order_details)
    
    # 지정가 및 TWAP 전략 가중치가 증가해야 함
    assert adjusted_weights['limit'] > original_weights['limit']
    assert adjusted_weights['twap'] > original_weights['twap']

@pytest.mark.asyncio
async def test_analyze_performance(optimizer):
    """성능 분석 테스트"""
    market_state = 'normal'
    strategy = 'twap'
    
    # 초기 상태 확인
    scores = optimizer._analyze_performance(market_state)
    assert scores[strategy] == 0.0
    
    # 데이터 추가
    optimizer.add_execution_result(strategy, market_state, 0.8)
    optimizer.add_execution_result(strategy, market_state, 0.7)
    optimizer.add_execution_result(strategy, market_state, 0.9)
    
    # 성능 분석
    scores = optimizer._analyze_performance(market_state)
    assert scores[strategy] > 0.0
    assert scores[strategy] <= 1.0
    
    # 최근 데이터에 더 큰 가중치 부여 확인
    # 마지막 데이터인 0.9가 더 큰 영향을 주어야 함
    assert scores[strategy] > np.mean([0.8, 0.7, 0.9])

@pytest.mark.asyncio
async def test_has_sufficient_data(optimizer):
    """충분한 데이터 확인 테스트"""
    market_state = 'normal'
    
    # 초기 상태
    assert optimizer._has_sufficient_data(market_state) == False
    
    # 일부 전략에만 데이터 추가
    optimizer.add_execution_result('twap', market_state, 0.8)
    optimizer.add_execution_result('twap', market_state, 0.7)
    optimizer.add_execution_result('twap', market_state, 0.9)
    
    # 여전히 충분하지 않음 (모든 전략에 충분한 데이터가 필요)
    assert optimizer._has_sufficient_data(market_state) == False
    
    # 모든 전략에 데이터 추가
    for strategy in optimizer.strategies:
        if strategy != 'twap':  # twap은 이미 추가됨
            optimizer.add_execution_result(strategy, market_state, 0.8)
            optimizer.add_execution_result(strategy, market_state, 0.7)
            optimizer.add_execution_result(strategy, market_state, 0.9)
    
    # 이제 충분함
    assert optimizer._has_sufficient_data(market_state) == True

@pytest.mark.asyncio
async def test_update_weights(optimizer):
    """가중치 업데이트 테스트"""
    market_state = 'normal'
    original_weights = optimizer.strategy_weights[market_state].copy()
    
    # 각 전략에 다른 점수 부여
    strategy_scores = {
        'twap': 0.9,
        'vwap': 0.8,
        'market': 0.6,
        'limit': 0.7,
        'iceberg': 0.5,
        'adaptive': 0.4
    }
    
    # 가중치 업데이트
    optimizer._update_weights(market_state, strategy_scores)
    
    # 업데이트된 가중치 확인
    updated_weights = optimizer.strategy_weights[market_state]
    
    # 점수가 높은 전략은 가중치가 높아야 함
    assert updated_weights['twap'] > updated_weights['adaptive']
    assert updated_weights['vwap'] > updated_weights['iceberg']
    
    # 가중치 합이 1이어야 함
    assert sum(updated_weights.values()) == pytest.approx(1.0)
    
    # 탐색 비율이 반영되어야 함 (모든 전략에 최소한 일정 가중치 부여)
    min_weight = optimizer.exploration_rate / len(optimizer.strategies)
    for strategy in optimizer.strategies:
        assert updated_weights[strategy] >= min_weight

@pytest.mark.asyncio
async def test_optimize_strategy_weights(optimizer):
    """전략 가중치 최적화 테스트"""
    market_state = 'normal'
    
    # 모든 전략에 충분한 데이터 제공
    for strategy in optimizer.strategies:
        # 각 전략마다 다른 성능 점수 부여
        if strategy == 'twap':
            scores = [0.9, 0.85, 0.95]
        elif strategy == 'vwap':
            scores = [0.8, 0.75, 0.85]
        elif strategy == 'market':
            scores = [0.6, 0.65, 0.55]
        elif strategy == 'limit':
            scores = [0.7, 0.75, 0.65]
        elif strategy == 'iceberg':
            scores = [0.5, 0.55, 0.45]
        else:  # adaptive
            scores = [0.4, 0.45, 0.35]
            
        for score in scores:
            optimizer.add_execution_result(strategy, market_state, score)
    
    # 최적화 전 가중치 저장
    before_weights = optimizer.strategy_weights[market_state].copy()
    
    # 전략 가중치 최적화
    await optimizer._optimize_strategy_weights()
    
    # 최적화 후 가중치 확인
    after_weights = optimizer.strategy_weights[market_state]
    
    # 가중치가 변경되었는지 확인
    assert before_weights != after_weights
    
    # 성능이 좋은 전략의 가중치가 증가했는지 확인
    assert after_weights['twap'] > after_weights['adaptive']
    assert after_weights['vwap'] > after_weights['iceberg']

@pytest.mark.asyncio
async def test_get_strategy_performance(optimizer):
    """전략 성능 통계 테스트"""
    market_state = 'normal'
    
    # 일부 전략에 데이터 추가
    optimizer.add_execution_result('twap', market_state, 0.8)
    optimizer.add_execution_result('twap', market_state, 0.9)
    
    # 성능 통계 조회
    stats = optimizer.get_strategy_performance(market_state)
    
    # 통계 확인
    assert market_state in stats
    assert 'twap' in stats[market_state]
    assert stats[market_state]['twap']['count'] == 2
    assert stats[market_state]['twap']['avg'] == pytest.approx(0.85)
    assert stats[market_state]['twap']['min'] == pytest.approx(0.8)
    assert stats[market_state]['twap']['max'] == pytest.approx(0.9)
    
    # 데이터가 없는 전략 확인
    assert 'vwap' in stats[market_state]
    assert stats[market_state]['vwap']['count'] == 0

@pytest.mark.asyncio
async def test_get_strategy_weights(optimizer):
    """전략 가중치 조회 테스트"""
    # 특정 시장 상태의 가중치 조회
    weights = optimizer.get_strategy_weights('normal')
    assert 'normal' in weights
    assert len(weights['normal']) == len(optimizer.strategies)
    
    # 모든 시장 상태의 가중치 조회
    weights = optimizer.get_strategy_weights()
    for state in optimizer.strategy_weights:
        assert state in weights
        
    # 알 수 없는 시장 상태
    weights = optimizer.get_strategy_weights('unknown_state')
    assert weights == {}

@pytest.mark.asyncio
async def test_set_strategies(optimizer):
    """전략 목록 설정 테스트"""
    # 기존 전략 목록 저장
    original_strategies = optimizer.strategies.copy()
    
    # 새 전략 목록
    new_strategies = ['twap', 'vwap', 'market', 'custom_strategy']
    
    # 전략 목록 설정
    optimizer.set_strategies(new_strategies)
    
    # 설정 확인
    assert optimizer.strategies == new_strategies
    
    # 모든 시장 상태에 새 전략 반영
    for state in optimizer.strategy_weights:
        for strategy in new_strategies:
            assert strategy in optimizer.strategy_weights[state]
            
    # 가중치 합이 1이어야 함
    for state in optimizer.strategy_weights:
        assert sum(optimizer.strategy_weights[state].values()) == pytest.approx(1.0)

@pytest.mark.asyncio
async def test_set_market_states(optimizer):
    """시장 상태 목록 설정 테스트"""
    # 새 시장 상태 목록
    new_states = ['normal', 'volatile', 'custom_state']
    
    # 시장 상태 목록 설정
    optimizer.set_market_states(new_states)
    
    # 설정 확인
    for state in new_states:
        assert state in optimizer.strategy_weights
        
    # 불필요한 시장 상태 제거 확인
    for state in optimizer.strategy_weights:
        assert state in new_states
        
    # 새 시장 상태 가중치 초기화 확인
    assert sum(optimizer.strategy_weights['custom_state'].values()) == pytest.approx(1.0)

@pytest.mark.asyncio
async def test_reset_weights(optimizer):
    """가중치 초기화 테스트"""
    market_state = 'normal'
    
    # 가중치 수정
    optimizer.manual_update_weight(market_state, 'twap', 0.5)
    
    # 수정 확인
    assert optimizer.strategy_weights[market_state]['twap'] == 0.5
    
    # 가중치 초기화
    optimizer.reset_weights(market_state)
    
    # 초기화 확인
    for strategy in optimizer.strategies:
        assert optimizer.strategy_weights[market_state][strategy] == pytest.approx(1.0 / len(optimizer.strategies))
        
    # 모든 상태 초기화
    optimizer.reset_weights()
    
    # 초기화 확인
    for state in optimizer.strategy_weights:
        for strategy in optimizer.strategies:
            assert optimizer.strategy_weights[state][strategy] == pytest.approx(1.0 / len(optimizer.strategies))

@pytest.mark.asyncio
async def test_manual_update_weight(optimizer):
    """가중치 수동 업데이트 테스트"""
    market_state = 'normal'
    strategy = 'twap'
    
    # 업데이트 전 다른 전략 가중치 저장
    other_weights_before = {
        s: optimizer.strategy_weights[market_state][s]
        for s in optimizer.strategies if s != strategy
    }
    
    # 가중치 수동 업데이트
    success = optimizer.manual_update_weight(market_state, strategy, 0.5)
    
    # 업데이트 확인
    assert success == True
    assert optimizer.strategy_weights[market_state][strategy] == 0.5
    
    # 다른 전략 가중치 합은 0.5가 되어야 함
    other_total = sum(optimizer.strategy_weights[market_state][s] for s in optimizer.strategies if s != strategy)
    assert other_total == pytest.approx(0.5)
    
    # 비율은 유지되어야 함
    if sum(other_weights_before.values()) > 0:
        for s in other_weights_before:
            assert optimizer.strategy_weights[market_state][s] / other_total == pytest.approx(
                other_weights_before[s] / sum(other_weights_before.values())
            )
    
    # 알 수 없는 시장 상태
    success = optimizer.manual_update_weight('unknown_state', strategy, 0.5)
    assert success == False
    
    # 알 수 없는 전략
    success = optimizer.manual_update_weight(market_state, 'unknown_strategy', 0.5)
    assert success == False
    
    # 음수 가중치
    success = optimizer.manual_update_weight(market_state, strategy, -0.1)
    assert success == False 