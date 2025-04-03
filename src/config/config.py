import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 거래소 설정
EXCHANGE = "binance"  # 사용할 거래소
SYMBOL = "BTC/USDT"   # 거래 페어

# API 설정
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# 거래 설정
LEVERAGE = 2  # 레버리지
POSITION_SIZE = 0.01  # 포지션 크기 (BTC)
STOP_LOSS_PERCENTAGE = 2.0  # 손절 비율 (%)
TAKE_PROFIT_PERCENTAGE = 4.0  # 익절 비율 (%)

# 전략 설정
TIMEFRAME = "1h"  # 시간 프레임
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# 알림 설정
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 데이터 저장 설정
DATA_DIR = "data_storage"
BACKTEST_RESULTS_DIR = "backtest_results" 