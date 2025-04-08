# 암호화폐 트레이딩 봇

암호화폐 시장을 위한 자동화된 트레이딩 시스템입니다.

## 주요 기능

- 실시간 시장 데이터 수집 및 분석
- 다양한 트레이딩 전략 지원
- 백테스팅 엔진
- 리스크 관리 시스템
- 웹 기반 대시보드
- 실시간 알림 시스템

## 설치 방법

### 요구사항

- Python 3.9 이상
- Docker 및 Docker Compose (선택사항)

### 로컬 설치

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
```bash
cp .env.example .env
# .env 파일을 편집하여 API 키 등 설정
```

### Docker를 사용한 설치

1. Docker 이미지 빌드:
```bash
docker-compose build
```

2. 서비스 시작:
```bash
docker-compose up -d
```

## 사용 방법

### 실시간 트레이딩

```bash
python run_trading.py
```

### 백테스팅

```bash
python run_backtest.py
```

### 웹 인터페이스

```bash
streamlit run run_web.py
```

## 프로젝트 구조

```
.
├── src/
│   ├── data/           # 데이터 수집 및 처리
│   ├── strategy/       # 트레이딩 전략
│   ├── risk/           # 리스크 관리
│   ├── trading/        # 트레이딩 시스템
│   ├── backtesting/    # 백테스팅 엔진
│   ├── web/            # 웹 인터페이스
│   └── utils/          # 유틸리티 함수
├── tests/              # 테스트 코드
├── config/             # 설정 파일
├── data/               # 데이터 저장
├── results/            # 결과 저장
└── logs/               # 로그 파일
```

## 설정

### API 키 설정

`.env` 파일에 다음 정보를 추가하세요:

```
API_KEY=your_api_key
API_SECRET=your_api_secret
EXCHANGE_ID=binance
SYMBOLS=BTC/USDT,ETH/USDT
```

### 전략 설정

`config/strategy.yaml` 파일에서 전략 파라미터를 설정할 수 있습니다.

## 테스트

```bash
pytest tests/
```

## 기여하기

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 라이센스

MIT License

## 문의

문제나 제안사항이 있으시면 이슈를 생성해주세요. 