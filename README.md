# 암호화폐 트레이딩 봇

암호화폐 시장에서 자동화된 트레이딩을 수행하는 봇입니다. 기술적 분석, 백테스팅, 최적화, 알림 시스템 등을 포함한 종합적인 트레이딩 솔루션을 제공합니다.

## 주요 기능

### 1. 기술적 분석
- RSI, MACD, 볼린저 밴드 등 다양한 기술적 지표 분석
- 실시간 시장 데이터 모니터링
- 차트 분석 및 시각화

### 2. 백테스팅
- 과거 데이터 기반 전략 테스트
- 다양한 성과 지표 분석
- 자본금 곡선 및 낙폭 분석

### 3. 전략 최적화
- 파라미터 그리드 서치
- 유전 알고리즘 기반 최적화
- 성과 메트릭스 분석

### 4. 알림 시스템
- 텔레그램 통합
- 사용자 정의 알림 규칙
- 실시간 거래 알림

### 5. 백업 및 복구
- 데이터베이스 백업
- 설정 파일 백업
- 로그 파일 백업
- 전략 파일 백업

### 6. 성능 모니터링
- CPU, 메모리, 디스크 사용량 모니터링
- 네트워크 I/O 모니터링
- 프로세스 및 스레드 모니터링
- 경고 시스템

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/hsh723/coin_trading.git
cd coin_trading
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 편집하여 API 키 등 설정
```

## 사용 방법

1. Streamlit 웹 인터페이스 실행
```bash
streamlit run streamlit_app.py
```

2. 웹 브라우저에서 `http://localhost:8501` 접속

## 프로젝트 구조

```
coin_trading/
├── src/
│   ├── analysis/
│   │   ├── technical_analyzer.py
│   │   └── self_learning.py
│   ├── strategy/
│   │   ├── base_strategy.py
│   │   └── portfolio_manager.py
│   ├── backtest/
│   │   ├── backtest_engine.py
│   │   └── backtest_analyzer.py
│   ├── dashboard/
│   │   └── dashboard.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── performance_monitor.py
│   │   └── data_loader.py
│   ├── api/
│   │   └── api_manager.py
│   ├── backup/
│   │   └── backup_manager.py
│   ├── optimization/
│   │   └── optimizer.py
│   ├── notification/
│   │   ├── telegram_notifier.py
│   │   └── notification_manager.py
│   ├── database/
│   │   └── database_manager.py
│   └── exchange/
│       └── binance_exchange.py
├── config/
│   └── config.yaml
├── data/
│   ├── market_data.csv
│   └── historical_data/
├── logs/
│   ├── app.log
│   └── performance.log
├── backup/
│   ├── database/
│   ├── config/
│   ├── logs/
│   └── strategies/
├── tests/
│   ├── test_technical_analyzer.py
│   ├── test_backtest_engine.py
│   └── test_performance_monitor.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
└── streamlit_app.py
```

## 주요 모듈 설명

### 1. 분석 모듈 (`src/analysis/`)
- `technical_analyzer.py`: 기술적 지표 계산 및 분석
- `self_learning.py`: 머신러닝 기반 시장 분석

### 2. 전략 모듈 (`src/strategy/`)
- `base_strategy.py`: 기본 전략 클래스
- `portfolio_manager.py`: 포트폴리오 관리

### 3. 백테스팅 모듈 (`src/backtest/`)
- `backtest_engine.py`: 백테스팅 엔진
- `backtest_analyzer.py`: 백테스팅 결과 분석

### 4. 유틸리티 모듈 (`src/utils/`)
- `config.py`: 설정 관리
- `logger.py`: 로깅 시스템
- `performance_monitor.py`: 시스템 성능 모니터링
- `data_loader.py`: 데이터 로딩 및 전처리

### 5. API 모듈 (`src/api/`)
- `api_manager.py`: 거래소 API 통합

### 6. 백업 모듈 (`src/backup/`)
- `backup_manager.py`: 데이터 백업 및 복구

### 7. 최적화 모듈 (`src/optimization/`)
- `optimizer.py`: 전략 파라미터 최적화

### 8. 알림 모듈 (`src/notification/`)
- `telegram_notifier.py`: 텔레그램 알림
- `notification_manager.py`: 알림 규칙 관리

### 9. 데이터베이스 모듈 (`src/database/`)
- `database_manager.py`: 데이터베이스 관리

### 10. 거래소 모듈 (`src/exchange/`)
- `binance_exchange.py`: 바이낸스 거래소 통합

### 11. 테스트 모듈 (`tests/`)
- `test_technical_analyzer.py`: 기술적 분석 테스트
- `test_backtest_engine.py`: 백테스팅 엔진 테스트
- `test_performance_monitor.py`: 성능 모니터링 테스트

### 12. 설정 파일 (`config/`)
- `config.yaml`: 애플리케이션 설정

### 13. 데이터 디렉토리 (`data/`)
- `market_data.csv`: 시장 데이터
- `historical_data/`: 과거 데이터 저장

### 14. 로그 디렉토리 (`logs/`)
- `app.log`: 애플리케이션 로그
- `performance.log`: 성능 모니터링 로그

### 15. 백업 디렉토리 (`backup/`)
- `database/`: 데이터베이스 백업
- `config/`: 설정 파일 백업
- `logs/`: 로그 파일 백업
- `strategies/`: 전략 파일 백업

## 라이센스

MIT License

## 기여 방법

1. 이슈 생성
2. 브랜치 생성
3. 변경사항 커밋
4. 풀 리퀘스트 생성 