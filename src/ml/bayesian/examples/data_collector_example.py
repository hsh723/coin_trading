import asyncio
import logging
import os
from datetime import datetime, timedelta
import pandas as pd

from ..data_collector import DataCollector

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """메인 실행 함수"""
    try:
        # 데이터 저장 디렉토리 생성
        save_dir = "./data"
        os.makedirs(save_dir, exist_ok=True)
        
        # 데이터 수집기 초기화
        collector = DataCollector(
            websocket_url="wss://stream.binance.com:9443/ws",
            rest_api_url="https://api.binance.com/api/v3",
            symbols=["BTCUSDT", "ETHUSDT"],
            save_dir=save_dir
        )
        
        # 과거 데이터 수집
        logger.info("과거 데이터 수집 시작")
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        
        for symbol in collector.symbols:
            df = await collector.get_historical_data(
                symbol=symbol,
                interval='1m',
                start_time=start_time,
                end_time=end_time
            )
            
            if not df.empty:
                filename = f"{save_dir}/{symbol}_historical_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"과거 데이터 저장 완료: {filename}")
        
        # 실시간 데이터 수집 시작
        logger.info("실시간 데이터 수집 시작")
        await collector.start()
        
        # 5분간 데이터 수집
        await asyncio.sleep(300)
        
        # 데이터 수집 중지
        await collector.stop()
        
        # 수집 메트릭스 출력
        metrics = collector.get_metrics()
        logger.info(f"수집 메트릭스: {metrics}")
        
    except Exception as e:
        logger.error(f"데이터 수집 중 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 