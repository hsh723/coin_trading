# Coin Trading System

암호화폐 자동 거래 시스템

## 프로젝트 구조

```
coin_Trading/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.py          # 설정 파일 관리
│   │   └── settings.py        # 환경 설정
│   │
│   ├── exchange/
│   │   ├── __init__.py
│   │   ├── binance_client.py  # 바이낸스 API 클라이언트
│   │   └── exchange_client.py # 거래소 클라이언트 인터페이스
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── execution_manager.py      # 주문 실행 관리자
│   │   ├── execution_monitor.py      # 주문 실행 모니터링
│   │   ├── execution_quality_monitor.py  # 실행 품질 모니터링
│   │   ├── execution_notifier.py     # 실행 알림
│   │   ├── error_handler.py          # 에러 처리
│   │   ├── asset_cache_manager.py    # 자산 캐시 관리
│   │   ├── performance_metrics_collector.py  # 성능 메트릭 수집
│   │   ├── execution_strategy_optimizer.py   # 실행 전략 최적화
│   │   ├── position_manager.py       # 포지션 관리
│   │   └── strategies/
│   │       ├── __init__.py
│   │       ├── base.py               # 기본 전략 클래스
│   │       ├── market.py             # 시장 주문 전략
│   │       ├── limit.py              # 지정가 주문 전략
│   │       ├── iceberg.py            # 아이스버그 주문 전략
│   │       └── vwap.py               # VWAP 주문 전략
│   │
│   ├── market/
│   │   ├── __init__.py
│   │   ├── market_state_monitor.py   # 시장 상태 모니터링
│   │   └── market_data_manager.py    # 시장 데이터 관리
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── risk_manager.py           # 리스크 관리
│   │   └── risk_metrics.py           # 리스크 메트릭
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                 # 로깅 유틸리티
│   │   └── helpers.py                # 헬퍼 함수
│   │
│   └── main.py                       # 메인 애플리케이션
│
├── tests/
│   ├── __init__.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_binance_integration.py  # 바이낸스 통합 테스트
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   └── test_execution_manager.py    # 실행 관리자 테스트
│   │
│   └── strategies/
│       ├── __init__.py
│       ├── test_market_strategy.py      # 시장 주문 전략 테스트
│       ├── test_limit_strategy.py       # 지정가 주문 전략 테스트
│       ├── test_iceberg_strategy.py     # 아이스버그 전략 테스트
│       └── test_vwap_strategy.py        # VWAP 전략 테스트
│
├── requirements.txt                    # 의존성 패키지
└── README.md                          # 프로젝트 설명
```

## 주요 컴포넌트 설명

### 1. 설정 관리 (config/)
- `config.py`: 애플리케이션 설정 관리
  - 설정 파일 로드 및 검증
  - 환경별 설정 관리
  - API 키 및 비밀 키 관리
- `settings.py`: 환경별 설정 값 정의
  - 개발/테스트/운영 환경 설정
  - API 엔드포인트 설정
  - 로깅 설정

### 2. 거래소 통신 (exchange/)
- `binance_client.py`: 바이낸스 API와의 통신 담당
  - 주문 생성/취소/조회
  - 계정 정보 조회
  - 시장 데이터 조회
  - 시간 동기화
- `exchange_client.py`: 거래소 클라이언트 인터페이스 정의
  - 공통 인터페이스 정의
  - 에러 처리
  - 재시도 로직

### 3. 주문 실행 (execution/)
- `execution_manager.py`: 주문 실행의 핵심 관리자
  - 주문 실행 전략 선택
  - 주문 실행 및 모니터링
  - 포지션 관리
  - 리스크 관리
- `execution_monitor.py`: 주문 실행 상태 모니터링
  - 주문 상태 추적
  - 실행 결과 분석
  - 알림 처리
- `execution_quality_monitor.py`: 실행 품질 모니터링
  - 실행 품질 메트릭 수집
  - 문제 감지 및 보고
  - 품질 점수 계산
- `position_manager.py`: 포지션 관리
  - 포지션 정보 추적
  - 포지션 조정
  - 리스크 관리

### 4. 거래 전략 (execution/strategies/)
- `base.py`: 기본 전략 클래스
  - 공통 인터페이스 정의
  - 기본 구현 제공
- `market.py`: 시장 주문 전략
  - 즉시 체결 주문
  - 시장 가격 기반 실행
- `limit.py`: 지정가 주문 전략
  - 지정가 주문 실행
  - 가격 조정 로직
- `iceberg.py`: 아이스버그 주문 전략
  - 대량 주문 분할 실행
  - 시장 영향 최소화
- `vwap.py`: VWAP 주문 전략
  - 거래량 가중 평균 가격 기반 실행
  - 시장 영향 최소화

### 5. 시장 데이터 (market/)
- `market_state_monitor.py`: 시장 상태 모니터링
  - 가격 변동 모니터링
  - 거래량 분석
  - 시장 상태 평가
- `market_data_manager.py`: 시장 데이터 관리
  - 실시간 데이터 수집
  - 데이터 캐싱
  - 데이터 분석

### 6. 리스크 관리 (risk/)
- `risk_manager.py`: 리스크 관리 및 제한
  - 포지션 리스크 관리
  - 주문 리스크 검증
  - 리스크 제한 설정
- `risk_metrics.py`: 리스크 메트릭 계산
  - 변동성 계산
  - 드로다운 분석
  - 리스크 지표 계산

### 7. 유틸리티 (utils/)
- `logger.py`: 로깅 설정 및 관리
  - 로그 레벨 설정
  - 로그 포맷 설정
  - 파일/콘솔 로깅
- `helpers.py`: 공통 유틸리티 함수
  - 시간 변환
  - 숫자 포맷팅
  - 에러 처리

### 8. 테스트 (tests/)
- `test_binance_integration.py`: 바이낸스 API 통합 테스트
  - API 연결 테스트
  - 주문 실행 테스트
  - 데이터 조회 테스트
- `test_execution_manager.py`: 실행 관리자 테스트
  - 주문 실행 테스트
  - 포지션 관리 테스트
  - 리스크 관리 테스트
- 전략별 테스트 파일
  - 각 전략의 동작 검증
  - 에러 처리 테스트
  - 성능 테스트

### 9. 메인 애플리케이션 (main.py)
- FastAPI 기반 웹 서버
- REST API 엔드포인트
- 비동기 처리
- 에러 처리
- 문서화

## 설치 및 실행

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. 환경 설정:
- `src/config/settings.py` 파일에서 환경 설정
- API 키 및 비밀 키 설정

3. 서버 실행:
```bash
python src/main.py
```

## API 문서

서버 실행 후 다음 URL에서 API 문서 확인 가능:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 테스트 실행

```bash
# 모든 테스트 실행
pytest

# 특정 테스트 실행
pytest tests/integration/test_binance_integration.py
pytest tests/execution/test_execution_manager.py
```

## 라이센스

MIT License