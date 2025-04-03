import os
from dotenv import load_dotenv
from src.exchange.binance_exchange import BinanceExchange
import pandas as pd
from datetime import datetime

def main():
    # API 키 직접 설정
    api_key = "NuQYIUnhQbxisT5CxOIqfTkjEp45RV0spU5Z1RSpKvr0nY9jpkpbKRZgIbJqeYzq"
    api_secret = "IhVkkCPKPipVsE9YYlcaqueFihp6AysWajXrfARATu8xxVwdiLtgXT0gYGBYPvTQ"
    
    print("\nAPI 키 확인:")
    print(f"API KEY 존재: {'예' if api_key else '아니오'}")
    print(f"API SECRET 존재: {'예' if api_secret else '아니오'}")
    
    try:
        # 바이낸스 거래소 초기화
        exchange = BinanceExchange(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True
        )
        
        # BTC/USDT 5분봉 데이터 조회
        symbol = 'BTC/USDT'
        timeframe = '5m'
        limit = 100
        
        print(f"\n{symbol} {timeframe} 차트 데이터 조회 중...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit)
        
        if ohlcv:
            # 데이터프레임으로 변환
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 데이터 출력
            print(f"\n최근 {len(df)}개의 캔들 데이터:")
            print(df.tail())
            
            # 기본 통계 정보
            print("\n기본 통계 정보:")
            print(f"시작 시간: {df['timestamp'].min()}")
            print(f"종료 시간: {df['timestamp'].max()}")
            print(f"평균 거래량: {df['volume'].mean():.2f}")
            print(f"최대 거래량: {df['volume'].max():.2f}")
            print(f"최소 거래량: {df['volume'].min():.2f}")
        else:
            print("데이터 조회 실패")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 