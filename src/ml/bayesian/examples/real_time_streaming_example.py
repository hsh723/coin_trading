import asyncio
import logging
import sys
import os
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 경로 추가하여 모듈 import 가능하게 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.bayesian.model_factory import BayesianModelFactory
from src.ml.bayesian.model_deployment import ModelDeployer
from src.ml.bayesian.real_time_streaming import RealTimeStreaming

async def main():
    # 모델 학습
    logger.info("모델 학습 중...")
    model = BayesianModelFactory.get_model(
        model_type="ar",
        ar_order=3,
        seasonality=True,
        num_seasons=24
    )
    
    # 모델 배포 시스템 초기화
    logger.info("모델 배포 시스템 초기화 중...")
    deployer = ModelDeployer(
        model_dir="./models",
        api_host="0.0.0.0",
        api_port=8000
    )
    
    # 실시간 스트리밍 시스템 초기화
    logger.info("실시간 스트리밍 시스템 초기화 중...")
    streaming = RealTimeStreaming(
        websocket_url="ws://localhost:8765",  # WebSocket 서버 URL
        kafka_bootstrap_servers=["localhost:9092"],  # Kafka 브로커 주소
        model_deployer=deployer,
        batch_size=100,
        window_size=1000
    )
    
    try:
        # 스트리밍 시스템 시작
        logger.info("스트리밍 시스템 시작 중...")
        await streaming.start()
        
    except KeyboardInterrupt:
        logger.info("스트리밍 시스템 중지 중...")
        streaming.stop()
        logger.info("스트리밍 시스템 중지 완료")

if __name__ == "__main__":
    asyncio.run(main()) 