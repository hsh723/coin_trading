# 사용자 가이드

## 1. 시작하기

### 1.1 설치
1. 저장소 클론
```bash
git clone https://github.com/yourusername/coin-trading-bot.git
cd coin-trading-bot
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

4. 설정 파일 생성
```bash
cp config.example.yaml config.yaml
```

### 1.2 설정
`config.yaml` 파일을 수정하여 다음 설정을 구성합니다:

```yaml
# 거래소 설정
exchange:
  name: "binance"  # 거래소 이름
  api_key: ""      # API 키
  api_secret: ""   # API 시크릿
  testnet: true    # 테스트넷 사용 여부

# 전략 설정
strategy:
  name: "integrated"  # 전략 이름
  timeframe: "1h"     # 타임프레임
  symbols: ["BTC/USDT", "ETH/USDT"]  # 거래 심볼
  parameters:         # 전략 파라미터
    ma_period: 20
    rsi_period: 14
    rsi_overbought: 70
    rsi_oversold: 30

# 리스크 관리 설정
risk:
  initial_capital: 10000.0  # 초기 자본금
  risk_per_trade: 0.02     # 거래당 리스크 비율
  max_positions: 5         # 최대 포지션 수
  daily_loss_limit: 0.05   # 일일 손실 한도
  max_drawdown: 0.15       # 최대 손실폭

# 알림 설정
notification:
  telegram:
    enabled: true
    token: ""     # 텔레그램 봇 토큰
    chat_id: ""   # 채팅 ID
  email:
    enabled: true
    sender: ""    # 발신자 이메일
    password: ""  # 이메일 비밀번호
    receiver: ""  # 수신자 이메일
```

## 2. 트레이딩 봇 사용하기

### 2.1 시작 및 중지
```python
from src.bot.trading_bot import TradingBot
import yaml

# 설정 파일 로드
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 트레이딩 봇 생성
bot = TradingBot(config)

# 봇 시작
bot.start()

# 봇 중지
bot.stop()
```

### 2.2 웹 인터페이스
1. 웹 서버 시작
```bash
streamlit run src/web/streamlit_app.py
```

2. 브라우저에서 접속
- `http://localhost:8501` 접속

3. 주요 기능
- 실시간 차트
- 포지션 관리
- 성과 대시보드
- 설정 변경

## 3. 백테스팅

### 3.1 백테스팅 실행
```python
from src.backtest.backtester import Backtester
import pandas as pd

# 과거 데이터 로드
data = pd.read_csv('data/historical_data.csv')

# 백테스터 생성
backtester = Backtester(initial_capital=10000.0)

# 백테스팅 실행
results = backtester.run(data)

# 결과 출력
print(f"총 수익률: {results['metrics']['total_return']:.2%}")
print(f"승률: {results['metrics']['win_rate']:.2%}")
print(f"최대 손실폭: {results['metrics']['max_drawdown']:.2%}")
```

### 3.2 파라미터 최적화
```python
# 파라미터 최적화 실행
optimized_params = backtester.optimize_parameters(data)

# 최적 파라미터 출력
print("최적 파라미터:")
for param, value in optimized_params['best_params'].items():
    print(f"{param}: {value}")
```

## 4. 알림 설정

### 4.1 텔레그램 알림
1. 텔레그램 봇 생성
- @BotFather와 대화
- `/newbot` 명령어로 새 봇 생성
- 토큰과 채팅 ID 저장

2. 설정 파일에 추가
```yaml
notification:
  telegram:
    enabled: true
    token: "YOUR_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
```

### 4.2 이메일 알림
1. 이메일 설정
- Gmail 사용 시 앱 비밀번호 생성
- 설정 파일에 추가

```yaml
notification:
  email:
    enabled: true
    sender: "your.email@gmail.com"
    password: "YOUR_APP_PASSWORD"
    receiver: "receiver@example.com"
```

## 5. 리스크 관리

### 5.1 포지션 사이징
- 거래당 리스크 비율 설정
- 손절매 가격에 따른 포지션 크기 자동 계산

### 5.2 손실 제한
- 일일 손실 한도 설정
- 최대 손실폭 설정
- 자동 포지션 정리

## 6. 문제 해결

### 6.1 일반적인 문제
1. API 연결 오류
- API 키 확인
- 인터넷 연결 확인
- 거래소 상태 확인

2. 백테스팅 오류
- 데이터 형식 확인
- 파라미터 범위 확인
- 메모리 사용량 확인

### 6.2 로그 확인
```bash
# 로그 파일 확인
tail -f logs/trading_bot.log
```

## 7. 보안

### 7.1 API 키 관리
- API 키는 암호화되어 저장
- 주기적인 API 키 교체 권장
- 마스터 키 안전 보관

### 7.2 세션 관리
- 세션 타임아웃: 1시간
- IP 주소 및 사용자 에이전트 기록
- 비정상 접속 감지

## 8. 성능 최적화

### 8.1 시스템 요구사항
- CPU: 2코어 이상
- 메모리: 4GB 이상
- 디스크: 10GB 이상

### 8.2 최적화 팁
- 데이터베이스 인덱싱
- 캐시 활용
- 비동기 처리 