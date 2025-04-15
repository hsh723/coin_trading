# 실행 전략 모듈

이 문서는 암호화폐 거래 실행 시스템의 다양한 실행 전략에 대해 설명합니다.

## 1. 개요

실행 전략은 주문을 시장에서 어떻게 실행할지를 결정하는 알고리즘입니다. 실행 전략의 선택은 거래 비용, 슬리피지, 시장 영향 등에 큰 영향을 미칩니다.

현재 시스템은 다음과 같은 실행 전략을 제공합니다:

- **TWAP (Time-Weighted Average Price)**: 시간 가중 평균 가격 기반 실행
- **VWAP (Volume-Weighted Average Price)**: 거래량 가중 평균 가격 기반 실행
- **적응형 (Adaptive)**: 시장 상황에 동적으로 대응하는 전략
- **시장가 (Market)**: 즉시 체결을 위한 시장가 주문 전략
- **지정가 (Limit)**: 지정한 가격에 체결을 위한 지정가 주문 전략
- **아이스버그 (Iceberg)**: 대형 주문을 작은 조각으로 분할 실행하는 전략

## 2. 전략 구조

모든 실행 전략은 `BaseExecutionStrategy` 추상 클래스를 상속하여 구현됩니다:

```python
class BaseExecutionStrategy(abc.ABC):
    @abc.abstractmethod
    async def execute(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """주문 실행"""
        pass
        
    @abc.abstractmethod
    async def cancel(self) -> bool:
        """실행 취소"""
        pass
```

## 3. 전략별 특징

### 3.1 TWAP 전략

시간 가중 평균 가격(Time-Weighted Average Price) 전략은 주문을 일정한 시간 간격으로 분할하여 실행합니다.

**주요 특징:**
- 주문을 동일한 시간 간격으로 균등하게 분할
- 시간대에 따른 가격 변동 위험 분산
- 랜덤 요소를 적용하여 패턴 예측을 어렵게 함

**설정 옵션:**
- `time_window`: 총 실행 시간(초)
- `slice_count`: 분할 횟수
- `random_factor`: 랜덤 변동 계수 (0.0 ~ 1.0)

**사용 예:**
```python
config = {
    'time_window': 3600,  # 1시간
    'slice_count': 10,    # 10분할
    'random_factor': 0.2  # ±20% 랜덤 변동
}
strategy = TwapExecutionStrategy(config)
```

### 3.2 VWAP 전략

거래량 가중 평균 가격(Volume-Weighted Average Price) 전략은 과거 거래량 패턴에 따라 주문을 분할하여 실행합니다.

**주요 특징:**
- 거래량이 많은 시간대에 더 큰 비중으로 주문 실행
- 시장 유동성에 따른 실행 최적화
- 추정 VWAP를 기준으로 실행 방법 조정

**설정 옵션:**
- `time_window`: 총 실행 시간(초)
- `interval_count`: 시간 간격 수
- `volume_profile`: 거래량 프로필 (없으면 자동 생성)
- `deviation_limit`: 추정 VWAP 대비 최대 허용 편차

**사용 예:**
```python
config = {
    'time_window': 3600,       # 1시간
    'interval_count': 12,      # 5분 간격
    'deviation_limit': 0.002   # 추정 VWAP의 ±0.2% 허용
}
strategy = VwapExecutionStrategy(config)
```

### 3.3 적응형 전략

적응형(Adaptive) 전략은 시장 상황을 실시간으로 모니터링하며 최적의 실행 방법을 동적으로 선택합니다.

**주요 특징:**
- 시장 변동성에 따른 실행 속도 조정
- 스프레드, 유동성, 추세 등 다양한 시장 요소 고려
- 적극적(aggressive), 중립적(neutral), 소극적(passive) 전략 사이 동적 전환

**설정 옵션:**
- `slippage_threshold`: 스프레드 임계값
- `volatility_threshold`: 변동성 임계값
- `max_participation_rate`: 최대 참여율
- `initial_participation_rate`: 초기 참여율
- `urgency_factor`: 긴급도 (0.0 ~ 1.0)

**사용 예:**
```python
config = {
    'slippage_threshold': 0.002,      # 0.2%
    'volatility_threshold': 0.01,     # 1%
    'max_participation_rate': 0.3,    # 최대 30%
    'initial_participation_rate': 0.1, # 초기 10%
    'urgency_factor': 0.5             # 중간 긴급도
}
strategy = AdaptiveExecutionStrategy(config)
```

## 4. 전략 팩토리

전략 인스턴스 생성과 관리를 위해 `ExecutionStrategyFactory` 클래스를 제공합니다.

**주요 기능:**
- 다양한 전략 인스턴스 생성 및 캐싱
- 사용자 정의 전략 등록 지원
- 리소스 정리 자동화

**사용 예:**
```python
# 팩토리 설정
factory_config = {
    'twap': { 'time_window': 1800, 'slice_count': 6 },
    'vwap': { 'time_window': 1800, 'interval_count': 6 },
    'adaptive': { 'urgency_factor': 0.5 }
}

# 팩토리 생성 및 전략 인스턴스 조회
factory = ExecutionStrategyFactory(factory_config)
strategy = factory.get_strategy('twap')

# 사용 가능한 전략 조회
available_strategies = factory.get_available_strategies()
```

## 5. 전략 최적화기와 통합

실행 전략 최적화기(ExecutionStrategyOptimizer)는 시장 상태와 과거 성능 데이터를 기반으로 최적의 전략을 선택합니다.

**통합 방법:**
```python
# 최적화기 설정
optimizer_config = {
    'strategies': ['twap', 'vwap', 'market', 'limit', 'iceberg', 'adaptive'],
    'optimization_interval': 24,  # 시간
    'min_samples': 10
}

# 시장 상태에 따른 최적 전략 선택
market_state = 'volatile'
optimal_strategy = optimizer.get_optimal_strategy(market_state)

# 주문 특성을 고려한 전략 선택
order_details = {
    'quantity': 10.0,
    'market_volume': 100.0,
    'urgent': True
}
optimal_strategy = optimizer.get_optimal_strategy(market_state, order_details)
```

## 6. 확장 방법

새로운 실행 전략을 추가하려면:

1. `BaseExecutionStrategy`를 상속하는 새 클래스 구현
2. `execute()` 및 `cancel()` 메서드 구현
3. `ExecutionStrategyFactory`에 새 전략 등록

**예시:**
```python
# 새 전략 구현
class MyCustomStrategy(BaseExecutionStrategy):
    async def execute(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        # 커스텀 로직 구현
        
    async def cancel(self) -> bool:
        # 취소 로직 구현

# 팩토리에 등록
factory.register_strategy('custom', MyCustomStrategy)
```

## 7. 성능 최적화 팁

- 대량 주문은 주로 VWAP 또는 적응형 전략 사용
- 급박한 시장에서는 시장가 또는 적응형 전략 사용
- 변동성이 낮은 시장에서는 TWAP 전략 고려
- 변동성이 높은 시장에서는 아이스버그 전략 고려
- 주문 크기가 시장 거래량의 5% 이상인 경우 참여율 낮게 설정 