import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from ..model_trainer import ModelTrainer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """메인 실행 함수"""
    try:
        # 데이터 디렉토리 확인
        data_dir = "./processed_data"
        if not os.path.exists(data_dir):
            logger.error(f"데이터 디렉토리가 존재하지 않음: {data_dir}")
            return
        
        # 모델 트레이너 초기화
        trainer = ModelTrainer(
            data_dir=data_dir,
            model_dir="./models",
            sequence_length=60,
            prediction_length=10
        )
        
        # 심볼 목록
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # 각 심볼에 대한 모델 학습 및 예측
        for symbol in symbols:
            logger.info(f"{symbol} 모델 학습 시작")
            
            # 모델 학습
            model = trainer.train_model(symbol)
            if model is None:
                logger.warning(f"{symbol} 모델 학습 실패")
                continue
            
            # 데이터 로드
            data_files = [f for f in os.listdir(data_dir) if f.startswith(symbol)]
            if not data_files:
                continue
            
            df = pd.read_csv(os.path.join(data_dir, data_files[-1]))
            
            # 예측 수행
            predictions = trainer.predict(model, df)
            if len(predictions) == 0:
                logger.warning(f"{symbol} 예측 실패")
                continue
            
            # 예측 결과 시각화
            plt.figure(figsize=(12, 6))
            plt.plot(df['price'].values[-len(predictions):], label='실제 가격')
            plt.plot(predictions, label='예측 가격')
            plt.title(f"{symbol} 가격 예측")
            plt.xlabel("시간")
            plt.ylabel("가격")
            plt.legend()
            plt.savefig(f"./plots/{symbol}_prediction.png")
            plt.close()
            
            logger.info(f"{symbol} 예측 결과 저장 완료")
        
        # 학습 메트릭스 출력
        metrics = trainer.get_metrics()
        logger.info(f"학습 메트릭스: {metrics}")
        
    except Exception as e:
        logger.error(f"모델 학습 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 