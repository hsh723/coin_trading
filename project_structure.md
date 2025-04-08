# 암호화폐 트레이딩 봇 프로젝트 구조

## 루트 디렉토리 구조
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
├── .env.example           # 환경 변수 예제
├── config.yaml            # 전역 설정 파일
├── requirements.txt       # 의존성 패키지 목록
├── setup.py               # 패키지 설치 스크립트
├── Dockerfile             # Docker 이미지 설정
└── docker-compose.yml     # Docker 컨테이너 구성
```

## src 디렉토리 상세 구조
```
src/
├── main.py                # 메인 실행 파일
├── run_backtest.py        # 백테스트 실행 파일
├── analysis/             # 기술적 분석
│   ├── technical.py      # 기술적 지표
│   ├── pattern.py        # 차트 패턴
│   ├── machine_learning.py # 머신러닝 분석
│   └── __init__.py
├── strategy/            # 트레이딩 전략
│   ├── base.py         # 기본 전략 클래스
│   ├── portfolio.py    # 포트폴리오 관리
│   └── __init__.py
├── exchange/           # 거래소 인터페이스
│   ├── base.py        # 기본 거래소 클래스
│   ├── binance.py     # 바이낸스 구현
│   └── __init__.py
├── risk/              # 리스크 관리
│   ├── manager.py     # 리스크 관리자
│   └── __init__.py
├── utils/            # 유틸리티 함수
│   ├── logger.py     # 로깅 설정
│   ├── config.py     # 설정 관리
│   ├── data_loader.py # 데이터 로딩
│   └── __init__.py
├── database/         # 데이터베이스 관리
│   ├── models.py    # 데이터 모델
│   └── __init__.py
├── backtest/        # 백테스팅 시스템
│   ├── engine.py   # 백테스트 엔진
│   ├── analyzer.py # 결과 분석
│   └── __init__.py
├── dashboard/      # 웹 대시보드
│   ├── app.py     # 대시보드 앱
│   └── __init__.py
├── monitoring/    # 모니터링 시스템
│   ├── performance.py # 성능 모니터링
│   ├── alerts.py     # 알림 시스템
│   └── __init__.py
└── api/          # API 관리
    ├── manager.py # API 관리자
    └── __init__.py
```

## 주요 컴포넌트 설명

### 1. 분석 시스템 (src/analysis/)
- `technical.py`: RSI, MACD, 볼린저 밴드 등 기술적 지표
- `pattern.py`: 차트 패턴 인식 및 분석
- `machine_learning.py`: 머신러닝 기반 시장 분석

### 2. 전략 시스템 (src/strategy/)
- `base.py`: 기본 전략 클래스
- `portfolio.py`: 포트폴리오 관리

### 3. 거래소 인터페이스 (src/exchange/)
- `base.py`: 거래소 연동을 위한 기본 인터페이스
- `binance.py`: 바이낸스 API 구현

### 4. 리스크 관리 (src/risk/)
- `manager.py`: 포지션 크기, 손절, 익절 등 리스크 관리

### 5. 백테스팅 시스템 (src/backtest/)
- `engine.py`: 과거 데이터를 사용한 전략 테스트
- `analyzer.py`: 백테스팅 결과 분석

### 6. 모니터링 시스템 (src/monitoring/)
- `performance.py`: 시스템 성능 모니터링
- `alerts.py`: 알림 시스템 (텔레그램 통합)

### 7. 데이터 관리 (src/database/)
- `models.py`: 데이터 모델 및 스키마

### 8. 웹 대시보드 (src/dashboard/)
- `app.py`: Streamlit 기반 웹 인터페이스

### 9. 유틸리티 (src/utils/)
- `logger.py`: 로깅 시스템
- `config.py`: 설정 관리
- `data_loader.py`: 데이터 로딩 및 캐싱

### 10. API 관리 (src/api/)
- `manager.py`: API 키 및 요청 관리

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

### 1.1 분석 모듈 (`src/analysis/`)
- `technical_analyzer.py`: 기술적 지표 계산 및 분석
  - RSI, MACD, 볼린저 밴드 등 기술적 지표 구현
  - 차트 패턴 인식 및 분석
  - 시장 추세 분석
- `self_learning.py`: 머신러닝 기반 시장 분석
  - 시계열 데이터 전처리
  - 모델 학습 및 예측
  - 성능 평가 및 최적화

### 1.2 전략 모듈 (`src/strategy/`)
- `base_strategy.py`: 기본 전략 클래스
  - 전략 인터페이스 정의
  - 공통 메서드 구현
  - 리스크 관리 로직
- `portfolio_manager.py`: 포트폴리오 관리
  - 자산 배분 관리
  - 리밸런싱 로직
  - 성과 추적

### 1.3 백테스팅 모듈 (`src/backtest/`)
- `backtest_engine.py`: 백테스팅 엔진
  - 과거 데이터 기반 시뮬레이션
  - 거래 실행 및 기록
  - 성과 지표 계산
- `backtest_analyzer.py`: 백테스팅 결과 분석
  - 수익률 분석
  - 리스크 지표 계산
  - 최적화 포인트 식별

### 1.4 대시보드 모듈 (`src/dashboard/`)
- `dashboard.py`: 웹 인터페이스
  - Streamlit 기반 UI 구현
  - 실시간 데이터 시각화
  - 사용자 상호작용 처리

### 1.5 유틸리티 모듈 (`src/utils/`)
- `config.py`: 설정 관리
  - YAML 설정 파일 처리
  - 환경 변수 관리
  - 설정 검증
- `logger.py`: 로깅 시스템
  - 로그 레벨 관리
  - 파일 및 콘솔 로깅
  - 로그 포맷팅
- `performance_monitor.py`: 시스템 성능 모니터링
  - CPU/메모리 사용량 모니터링
  - 디스크/네트워크 I/O 모니터링
  - 경고 시스템
- `data_loader.py`: 데이터 로딩 및 전처리
  - CSV/JSON 데이터 로딩
  - 데이터 정제 및 변환
  - 캐싱 메커니즘

### 1.6 API 모듈 (`src/api/`)
- `api_manager.py`: 거래소 API 통합
  - API 키 관리
  - 요청/응답 처리
  - 에러 핸들링

### 1.7 백업 모듈 (`src/backup/`)
- `backup_manager.py`: 데이터 백업 및 복구
  - 증분 백업
  - 복구 프로세스
  - 백업 검증

### 1.8 최적화 모듈 (`src/optimization/`)
- `optimizer.py`: 전략 파라미터 최적화
  - 그리드 서치
  - 유전 알고리즘
  - 성과 메트릭스

### 1.9 알림 모듈 (`src/notification/`)
- `telegram_notifier.py`: 텔레그램 알림
  - 메시지 포맷팅
  - 알림 전송
  - 상태 모니터링
- `notification_manager.py`: 알림 규칙 관리
  - 규칙 정의
  - 조건 평가
  - 알림 스케줄링

### 1.10 데이터베이스 모듈 (`src/database/`)
- `database_manager.py`: 데이터베이스 관리
  - 연결 관리
  - 쿼리 실행
  - 트랜잭션 처리

### 1.11 거래소 모듈 (`src/exchange/`)
- `binance_exchange.py`: 바이낸스 거래소 통합
  - 주문 처리
  - 잔고 조회
  - 시장 데이터 수집

## 2. 설정 및 데이터 (`config/`, `data/`)

### 2.1 설정 파일 (`config/`)
- `config.yaml`: 애플리케이션 설정
  - API 설정
  - 전략 파라미터
  - 시스템 설정

### 2.2 데이터 디렉토리 (`data/`)
- `market_data.csv`: 시장 데이터
  - 실시간 가격 데이터
  - 거래량 데이터
  - OHLCV 데이터
- `historical_data/`: 과거 데이터 저장
  - 분봉 데이터
  - 일봉 데이터
  - 주봉 데이터

## 3. 로그 및 백업 (`logs/`, `backup/`)

### 3.1 로그 디렉토리 (`logs/`)
- `app.log`: 애플리케이션 로그
  - 거래 기록
  - 시스템 이벤트
  - 에러 로그
- `performance.log`: 성능 모니터링 로그
  - 시스템 메트릭스
  - 리소스 사용량
  - 경고 로그

### 3.2 백업 디렉토리 (`backup/`)
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

## 5. 루트 디렉토리 파일

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