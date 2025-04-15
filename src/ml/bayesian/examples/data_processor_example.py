import logging
import os
import pandas as pd
from datetime import datetime

from ..data_processor import DataProcessor

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """메인 실행 함수"""
    try:
        # 데이터 디렉토리 확인
        data_dir = "./data"
        if not os.path.exists(data_dir):
            logger.error(f"데이터 디렉토리가 존재하지 않음: {data_dir}")
            return
        
        # 데이터 프로세서 초기화
        processor = DataProcessor(
            data_dir=data_dir,
            save_dir="./processed_data"
        )
        
        # 심볼 목록
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # 각 심볼에 대한 데이터 처리
        for symbol in symbols:
            logger.info(f"{symbol} 데이터 처리 시작")
            
            # 데이터 처리
            df = processor.process_data(symbol)
            
            if not df.empty:
                # 처리된 데이터 확인
                logger.info(f"처리된 데이터 샘플:")
                logger.info(df.head())
                
                # 기술적 지표 확인
                logger.info("기술적 지표 통계:")
                logger.info(df[['MA5', 'MA20', 'MA60', 'RSI', 'MACD']].describe())
                
                # 시계열 특성 확인
                logger.info("시계열 특성 통계:")
                logger.info(df[['returns', 'volatility', 'price_momentum']].describe())
            
            else:
                logger.warning(f"{symbol} 데이터 처리 실패")
        
        # 처리 메트릭스 출력
        metrics = processor.get_metrics()
        logger.info(f"처리 메트릭스: {metrics}")
        
    except Exception as e:
        logger.error(f"데이터 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 