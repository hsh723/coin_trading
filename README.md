# 통합 트레이딩 시스템

이 프로젝트는 여러 기술적 지표와 패턴을 통합하여 암호화폐 트레이딩을 자동화하는 시스템입니다.

## 주요 특징

### 1. 통합된 기술적 분석
- 이동평균선 (20일, 60일, 200일)
- 볼린저 밴드
- RSI
- 스토캐스틱
- ABC 패턴
- 추세선 분석
- 캔들 패턴 분석

### 2. 정교한 진입 조건
- 추세선 터치/근접 판단 (0.2% 이내)
- 볼린저 밴드 터치/돌파 판단
- RSI 과매수/과매도
- 캔들 패턴 분석
- 거래량 분석
- 가격 모멘텀 분석

### 3. 리스크 관리
- 첫 진입은 전체 포지션의 50%만 사용
- 추가 진입은 B 포인트 돌파 시
- 손절/익절 15%
- 최대 보유 시간 2시간
- 추세 반전 시 청산

### 4. 신호 강도 시스템
- 추세선 터치: 0.3점
- 볼린저 밴드 터치: 0.2점
- RSI 조건: 0.2점
- 가격 행동: 각각 0.1점씩

## 설치 방법

1. 저장소 클론:
```bash
git clone [repository-url]
cd coin-trading
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
`.env` 파일을 생성하고 다음 내용을 추가:
```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

## 사용 방법

1. 트레이더 실행:
```bash
python src/run_integrated_trader.py
```

2. 백테스팅 실행:
```bash
python src/backtest_integrated_strategy.py
```

## 프로젝트 구조

```
coin-trading/
├── src/
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── integrated_strategy.py
│   ├── traders/
│   │   ├── __init__.py
│   │   └── integrated_trader.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── risk_manager.py
│   ├── run_integrated_trader.py
│   └── backtest_integrated_strategy.py
├── tests/
│   ├── __init__.py
│   ├── test_strategy.py
│   └── test_trader.py
├── logs/
├── .env
├── requirements.txt
└── README.md
```

## 로깅

모든 거래 기록은 `logs/integrated_trader.log`에 저장됩니다:
- 진입/청산 시점
- 진입/청산 사유
- 포지션 크기
- 수익/손실
- 신호 강도

## 주의사항

- 실제 거래 전에 반드시 백테스팅을 수행하세요.
- API 키는 안전하게 보관하세요.
- 리스크 관리 규칙을 준수하세요.
- 시장 상황에 따라 전략을 조정하세요. 