import os
from dotenv import load_dotenv
from traders.integrated_trader import IntegratedTrader

def main():
    # 환경 변수 로드
    load_dotenv()
    
    # 거래소 설정
    exchange = 'binance'
    symbol = 'BTC/USDT'
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    # 트레이더 초기화
    trader = IntegratedTrader(
        exchange=exchange,
        symbol=symbol,
        api_key=api_key,
        api_secret=api_secret
    )
    
    try:
        # 트레이더 실행
        trader.run()
    except KeyboardInterrupt:
        print("\n트레이더를 종료합니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main() 