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
├── strategies/            # 트레이딩 전략
│   ├── base.py           # 기본 전략 클래스
│   ├── integrated.py     # 통합 전략
│   └── __init__.py
├── indicators/           # 기술적 지표
│   ├── basic.py         # 기본 지표
│   ├── advanced.py      # 고급 지표
│   └── __init__.py
├── exchange/            # 거래소 인터페이스
│   ├── base.py         # 기본 거래소 클래스
│   ├── binance.py      # 바이낸스 구현
│   └── __init__.py
├── risk/               # 리스크 관리
│   ├── manager.py     # 리스크 관리자
│   └── __init__.py
├── utils/             # 유틸리티 함수
│   ├── logger.py     # 로깅 설정
│   ├── config.py     # 설정 관리
│   └── __init__.py
├── database/         # 데이터베이스 관리
│   ├── models.py    # 데이터 모델
│   └── __init__.py
├── backtest/        # 백테스팅 시스템
│   ├── engine.py   # 백테스트 엔진
│   └── __init__.py
├── dashboard/      # 웹 대시보드
│   ├── app.py     # 대시보드 앱
│   └── __init__.py
├── monitoring/    # 모니터링 시스템
│   ├── alerts.py # 알림 시스템
│   └── __init__.py
└── notification/ # 알림 시스템
    ├── telegram.py # 텔레그램 알림
    └── __init__.py
```

## 주요 컴포넌트 설명

### 1. 트레이딩 시스템 (src/strategies/)
- `base.py`: 기본 전략 클래스 정의, 모든 전략의 기본 구조 제공
- `integrated.py`: 여러 전략을 통합한 복합 전략 구현

### 2. 기술적 지표 (src/indicators/)
- `basic.py`: RSI, MACD, 이동평균 등 기본적인 기술 지표
- `advanced.py`: 고급 기술적 지표 및 커스텀 지표

### 3. 거래소 인터페이스 (src/exchange/)
- `base.py`: 거래소 연동을 위한 기본 인터페이스
- `binance.py`: 바이낸스 API 구현

### 4. 리스크 관리 (src/risk/)
- `manager.py`: 포지션 크기, 손절, 익절 등 리스크 관리 로직

### 5. 백테스팅 시스템 (src/backtest/)
- `engine.py`: 과거 데이터를 사용한 전략 테스트
- 성능 분석 및 최적화 도구

### 6. 모니터링 및 알림 (src/monitoring/, src/notification/)
- 실시간 성능 모니터링
- 텔레그램을 통한 알림 시스템

### 7. 데이터 관리 (src/database/)
- 거래 기록, 시장 데이터 저장
- 데이터 분석을 위한 구조화

### 8. 웹 대시보드 (src/dashboard/)
- 실시간 트레이딩 현황 시각화
- 성능 지표 및 포트폴리오 관리

### 9. 유틸리티 (src/utils/)
- 로깅 시스템
- 설정 관리
- 공통 유틸리티 함수

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