# 암호화폐 트레이딩 봇 프로젝트 구조

## 루트 디렉터리 구조
```
coin_Trading/
├── src/                    # 소스 코드 메인 디렉토리
├── config/                 # 설정 파일 디렉토리
├── data/                   # 데이터 저장소
├── logs/                   # 로그 파일 디렉토리
├── tests/                  # 테스트 코드
├── docs/                   # 문서화
├── notebooks/              # Jupyter 노트북
├── scripts/                # 유틸리티 스크립트
├── venv/                   # 가상 환경
├── .env                    # 환경 변수 설정
├── .env.example            # 환경 변수 예제
├── config.yaml             # 전역 설정 파일
├── requirements.txt        # 의존성 패키지 목록
├── setup.py                # 패키지 설치 스크립트
├── Dockerfile              # Docker 이미지 설정
└── docker-compose.yml      # Docker 컨테이너 구성
```

## 시스템 아키텍처 다이어그램

![System Architecture](docs/diagrams/system_architecture.png)

## src 디렉터리 상세 구조
```
src/
├── backtest/                # 백테스팅 시스템
│   ├── metrics_calculator.py
│   ├── position_tracker.py
│   ├── results_analyzer.py
│   ├── strategy_optimizer.py
│   ├── data_handler.py
│   ├── performance_analyzer.py
│   ├── trade_simulator.py
│   └── risk_evaluator.py
├── execution/               # 실행 시스템
│   ├── strategies/         # 주문 실행 전략
│   │   ├── base_strategy.py
│   │   ├── twap_strategy.py
│   │   ├── vwap_strategy.py
│   │   ├── adaptive_strategy.py
│   │   └── strategy_factory.py
│   ├── examples/           # 전략 사용 예제
│   ├── order_manager.py
│   ├── smart_routing.py
│   ├── trade_executor.py
│   ├── cost_analyzer.py
│   ├── slippage_monitor.py
│   └── market_impact.py
├── analytics/              # 분석 시스템
│   ├── market/
│   ├── volume/
│   ├── correlation/
│   └── risk/
├── data/                   # 데이터 파이프라인
│   ├── collectors/        # 데이터 수집
│   ├── processors/        # 데이터 처리
│   ├── storage/          # 데이터 저장
│   ├── validation/       # 데이터 검증
│   └── pipeline/         # 파이프라인 관리
├── alerts/                # 알림 시스템
│   ├── handlers/         # 알림 처리
│   ├── channels/         # 알림 채널
│   ├── filters/          # 알림 필터
│   └── templates/        # 알림 템플릿
└── real_time/             # 실시간 처리
    ├── market_data/
    ├── order_flow/
    └── execution/
```

## 주요 컴포넌트 설명

### 백테스팅 시스템
- 성과 분석 및 리포팅
- 전략 최적화
- 리스크 평가

### 실행 시스템
- 스마트 주문 실행
- 비용 최적화
- 실시간 모니터링
- 주문 실행 전략:
  - TWAP: 시간 가중 평균 가격 전략
  - VWAP: 거래량 가중 평균 가격 전략
  - Adaptive: 시장 상황에 따라 동적으로 조정되는 전략

### 분석 시스템
- 시장 분석
- 거래량 분석
- 상관관계 분석

### 데이터 파이프라인 시스템
- 실시간 데이터 수집
- 데이터 정제 및 변환
- 스트림 처리
- 데이터 저장 및 캐싱

### 알림 시스템
- 알림 우선순위 관리
- 다중 채널 통합
- 알림 에스컬레이션
- 알림 템플릿 관리

### 실시간 처리
- 실시간 시장 데이터 처리
- 주문 흐름 관리
- 실행 모니터링

## 데이터 흐름 다이어그램

![Data Flow](docs/diagrams/data_flow.png)

## 클래스 다이어그램

![Class Diagram](docs/diagrams/class_diagram.png)

## 설정 파일
- `.env`: API 키, 시크릿 등 민감한 설정
- `config.yaml`: 전략 파라미터, 거래 설정 등
- `requirements.txt`: 필요한 Python 패키지 목록

## 배포 관련 파일
- `Dockerfile`: 컨테이너 이미지 빌드 설정
- `docker-compose.yml`: 멀티 컨테이너 구성
- `run_trading.bat`: Windows 실행 스크립트

## 테스트 및 문서화
- `tests/`: 단위 테스트 및 통합 테스트
- `docs/`: API 문서 및 사용자 가이드
- `notebooks/`: 분석 및 전략 개발 노트북

# 프로젝트 구조 상세 설명

## 1. 소스 코드 구조 (`src/`)

### 1.1 백테스팅 모듈 (`src/backtest/`)
- `metrics_calculator.py`: 성과 지표 계산
- `position_tracker.py`: 포지션 추적
- `results_analyzer.py`: 결과 분석
- `strategy_optimizer.py`: 전략 최적화
- `data_handler.py`: 데이터 처리
- `performance_analyzer.py`: 성과 분석
- `trade_simulator.py`: 거래 시뮬레이션
- `risk_evaluator.py`: 리스크 평가

### 1.2 실행 모듈 (`src/execution/`)
- `strategies/`: 주문 실행 전략 모음
  - `base_strategy.py`: 모든 전략의 기본 클래스
  - `twap_strategy.py`: 시간 가중 평균 가격 전략
  - `vwap_strategy.py`: 거래량 가중 평균 가격 전략
  - `adaptive_strategy.py`: 적응형 실행 전략
  - `strategy_factory.py`: 전략 생성 팩토리
- `examples/`: 전략 사용 예제
- `order_manager.py`: 주문 관리
- `smart_routing.py`: 스마트 라우팅
- `trade_executor.py`: 거래 실행
- `cost_analyzer.py`: 비용 분석
- `slippage_monitor.py`: 슬리피지 모니터링
- `market_impact.py`: 시장 영향 분석
- `execution_manager.py`: 실행 관리자
- `strategy_optimizer.py`: 전략 최적화
- `real_time_monitor.py`: 실시간 모니터링
- `execution_quality_monitor.py`: 실행 품질 모니터링

### 1.3 분석 모듈 (`src/analytics/`)
- `market/`: 시장 분석
- `volume/`: 거래량 분석
- `correlation/`: 상관관계 분석
- `risk/`: 리스크 분석

### 1.4 데이터 파이프라인 모듈 (`src/data/`)
- `collectors/`: 데이터 수집
- `processors/`: 데이터 처리
- `storage/`: 데이터 저장
- `validation/`: 데이터 검증
- `pipeline/`: 파이프라인 관리

### 1.5 알림 모듈 (`src/alerts/`)
- `handlers/`: 알림 처리
- `channels/`: 알림 채널
- `filters/`: 알림 필터
- `templates/`: 알림 템플릿

### 1.6 실시간 처리 모듈 (`src/real_time/`)
- `market_data/`: 실시간 시장 데이터
- `order_flow/`: 주문 흐름 관리
- `execution/`: 실행 모니터링

## 2. 설정 및 데이터 (`config/`, `data/`)

### 2.1 설정 파일 (`config/`)
- `config.yaml`: 애플리케이션 설정
  - API 설정
  - 전략 파라미터
  - 시스템 설정

### 2.2 데이터 디렉터리 (`data/`)
- `market_data.csv`: 시장 데이터
  - 실시간 가격 데이터
  - 거래량 데이터
  - OHLCV 데이터
- `historical_data/`: 과거 데이터 저장
  - 분봉 데이터
  - 일봉 데이터
  - 주봉 데이터

## 3. 로그 및 백업 (`logs/`, `backup/`)

### 3.1 로그 디렉터리 (`logs/`)
- `app.log`: 애플리케이션 로그
  - 거래 기록
  - 시스템 이벤트
  - 에러 로그
- `performance.log`: 성능 모니터링 로그
  - 시스템 메트릭스
  - 리소스 사용량
  - 경고 로그

### 3.2 백업 디렉터리 (`backup/`)
- `database/`: 데이터베이스 백업
  - 전체 백업
  - 증분 백업
- `config/`: 설정 파일 백업
  - 설정 스냅샷
  - 버전 관리
- `logs/`: 로그 파일 백업
  - 로그 아카이브
  - 압축 저장
- `strategies/`: 전략 파일 백업
  - 전략 코드
  - 파라미터 설정

## 4. 테스트 (`tests/`)

### 4.1 테스트 모듈
- `test_technical_analyzer.py`: 기술적 분석 테스트
  - 지표 계산 검증
  - 패턴 인식 테스트
  - 성능 벤치마크
- `test_backtest_engine.py`: 백테스팅 엔진 테스트
  - 시뮬레이션 검증
  - 성과 지표 테스트
  - 엣지 케이스 처리
- `test_performance_monitor.py`: 성능 모니터링 테스트
  - 메트릭스 수집 테스트
  - 경고 시스템 검증
  - 리소스 사용 모니터링

## 5. 루트 디렉터리 파일

### 5.1 설정 파일
- `requirements.txt`: Python 패키지 의존성
- `.env`: 환경 변수 설정
- `.env.example`: 환경 변수 템플릿
- `.gitignore`: Git 무시 파일 설정

### 5.2 실행 파일
- `streamlit_app.py`: 메인 애플리케이션
  - 웹 인터페이스
  - 모듈 통합
  - 이벤트 루프 관리

# 최근 업데이트된 컴포넌트

## 1. 주문 실행 전략 (`src/execution/strategies/`)

### 1.1 기본 전략 클래스 (`base_strategy.py`)
- 모든 실행 전략의 추상 기본 클래스
- 표준 인터페이스 정의 (execute, cancel 메서드)
- 공통 유틸리티 및 로깅 기능

### 1.2 TWAP 전략 (`twap_strategy.py`)
- 시간 가중 평균 가격 기반 주문 실행 전략
- 주문을 일정 시간 간격으로 분할 실행
- 시간대에 따른 가격 변동 위험 분산
- 랜덤 요소를 적용하여 패턴 예측 방지

### 1.3 VWAP 전략 (`vwap_strategy.py`)
- 거래량 가중 평균 가격 기반 주문 실행 전략
- 과거 거래량 프로필에 따라 주문 분할
- 거래량이 많은 시간대에 더 큰 비중 할당
- 유동성 최적화를 위한 실행 조정

### 1.4 적응형 전략 (`adaptive_strategy.py`)
- 시장 상황에 실시간으로 대응하는 동적 전략
- 변동성, 스프레드, 유동성에 따른 실행 방법 조정
- 시장 상황에 따른 적극/중립/소극적 전략 간 전환
- 다양한 시장 지표 모니터링 및 분석

### 1.5 전략 팩토리 (`strategy_factory.py`)
- 다양한 실행 전략 인스턴스 생성 및 관리
- 전략 설정 관리 및 사용자 정의 전략 등록
- 전략 인스턴스 캐싱 및 자원 정리
- 사용 가능한 전략 목록 제공

## 2. 전략 예제 및 문서화 (`src/execution/examples/`)
- 각 실행 전략 사용 예제
- 다양한 시장 상황에 따른 전략 선택 가이드
- 실제 거래 환경에서의 최적화 방법 설명
- 성능 벤치마크 결과