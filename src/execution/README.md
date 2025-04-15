# 암호화폐 거래 실행 시스템

이 디렉토리에는 암호화폐 거래 실행 시스템의 핵심 구성 요소가 포함되어 있습니다. 

## 주요 구성 요소

### 실행 관리 모듈
- **ExecutionManager**: 전체 실행 시스템을 관리하는 중앙 컴포넌트
- **AssetCacheManager**: 자산 데이터 캐싱 및 관리
- **ExecutionStrategyOptimizer**: 시장 상태에 따른 최적 실행 전략 선택

### 모니터링 모듈
- **MarketStateMonitor**: 시장 상태 실시간 모니터링
- **ExecutionMonitor**: 주문 실행 성능 모니터링
- **ExecutionQualityMonitor**: 실행 품질 측정 및 모니터링
- **PerformanceMetricsCollector**: 실시간 성능 메트릭 수집

### 지원 모듈
- **ErrorHandler**: 실행 오류 관리
- **ExecutionNotifier**: 실행 이벤트 알림
- **ExecutionLogger**: 실행 데이터 로깅

## 실행 전략

시스템은 다음과 같은 실행 전략을 지원합니다:

1. **TWAP (Time Weighted Average Price)**: 시간 가중 평균 가격 기반 실행
2. **VWAP (Volume Weighted Average Price)**: 거래량 가중 평균 가격 기반 실행
3. **시장가 (Market)**: 즉시 체결을 위한 시장가 주문
4. **지정가 (Limit)**: 특정 가격에 체결을 위한 지정가 주문
5. **아이스버그 (Iceberg)**: 대량 주문을 작은 조각으로 분할 실행
6. **적응형 (Adaptive)**: 시장 조건에 따라 전략을 동적으로 조정

## 전략 최적화

`ExecutionStrategyOptimizer`는 다음과 같은 시장 상태에 따라 최적의 전략을 선택합니다:

- **normal**: 정상 시장 상태
- **volatile**: 변동성이 높은 시장
- **trending**: 추세가 강한 시장
- **illiquid**: 유동성이 낮은 시장
- **ranging**: 레인지 시장

최적화 과정은 다음 요소를 고려합니다:
- 과거 성능 메트릭
- 체결률 (Fill Rate)
- 슬리피지 (Slippage)
- 실행 비용 (Execution Cost)
- 주문 특성 (크기, 긴급성 등)

## 자산 캐시 관리

`AssetCacheManager`는 다음 데이터를 관리합니다:
- 실시간 가격
- 오더북
- 최근 체결 내역
- 자산 정보

캐시는 시간 제한 방식으로 관리되며, 주기적으로 갱신됩니다.

## 사용 방법

### 초기화 예제

```python
# 설정 정보
config = {
    'default_strategy': 'vwap',
    'max_retries': 3,
    'strategy_optimizer': {
        'strategies': ['twap', 'vwap', 'market', 'limit', 'iceberg', 'adaptive'],
        'optimization_interval': 24,  # 시간
        'min_samples': 10
    },
    'asset_cache': {
        'cache_ttl': 60,  # 초
        'refresh_interval': 5,  # 초
        'default_symbols': ['BTC/USDT', 'ETH/USDT']
    }
}

# 실행 매니저 초기화
execution_manager = ExecutionManager(config)
await execution_manager.initialize()

try:
    # 주문 실행
    order_result = await execution_manager.execute_order({
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'type': 'limit',
        'quantity': 0.1,
        'price': 50000.0
    })
    
    print(f"주문 실행 결과: {order_result}")
    
finally:
    # 리소스 정리
    await execution_manager.close()
```

### 전략 최적화 사용 예제

```python
# 전략 최적화기 설정
optimizer_config = {
    'strategies': ['twap', 'vwap', 'market', 'limit', 'iceberg', 'adaptive'],
    'optimization_interval': 24,  # 시간
    'min_samples': 10
}

# 전략 최적화기 초기화
optimizer = ExecutionStrategyOptimizer(optimizer_config)
await optimizer.initialize()

try:
    # 최적 전략 선택
    market_state = 'volatile'
    strategy = optimizer.get_optimal_strategy(market_state)
    
    print(f"시장 상태 '{market_state}'에 대한 최적 전략: {strategy}")
    
    # 주문 상세 정보 포함
    order_details = {
        'quantity': 10.0,
        'market_volume': 100.0,  # 10%
        'urgent': True
    }
    
    strategy = optimizer.get_optimal_strategy(market_state, order_details)
    print(f"주문 상세 정보를 고려한 최적 전략: {strategy}")
    
finally:
    # 리소스 정리
    await optimizer.close()
```

## 확장성

이 시스템은 새로운 전략이나 모듈을 쉽게 추가할 수 있도록 설계되었습니다:

1. 새로운 전략을 `ExecutionManager`에 등록하여 사용 가능
2. 사용자 정의 시장 상태를 `MarketStateMonitor`에 추가 가능
3. 다양한 성능 메트릭을 `PerformanceMetricsCollector`에 확장 가능 