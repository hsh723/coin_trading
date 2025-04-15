import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import pymc3 as pm
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 경로 추가하여 모듈 import 가능하게 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.bayesian.model_factory import BayesianModelFactory
from src.ml.bayesian.model_deployment import ModelDeployer

def generate_sample_data(n_samples: int = 1000) -> pd.Series:
    """샘플 시계열 데이터 생성"""
    np.random.seed(42)
    t = np.arange(n_samples)
    trend = 0.1 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, 1, n_samples)
    data = trend + seasonal + noise
    return pd.Series(data, index=pd.date_range(start='2023-01-01', periods=n_samples, freq='H'))

def main():
    # 샘플 데이터 생성
    logger.info("샘플 데이터 생성 중...")
    data = generate_sample_data()
    
    # 모델 학습
    logger.info("모델 학습 중...")
    model = BayesianModelFactory.get_model(
        model_type="ar",
        ar_order=3,
        seasonality=True,
        num_seasons=24
    )
    model.fit(data)
    
    # 모델 배포 시스템 초기화
    logger.info("모델 배포 시스템 초기화 중...")
    deployer = ModelDeployer(
        model_dir="./models",
        api_host="0.0.0.0",
        api_port=8000,
        monitoring_interval=60
    )
    
    # 모델 저장
    logger.info("모델 저장 중...")
    metadata = {
        "created_at": datetime.now().isoformat(),
        "model_type": "ar",
        "ar_order": 3,
        "seasonality": True,
        "num_seasons": 24,
        "training_data_size": len(data),
        "performance": {
            "rmse": 0.5,  # TODO: 실제 성능 지표 계산
            "mae": 0.4
        }
    }
    version = deployer.save_model(model, "price_predictor", metadata)
    logger.info(f"모델 저장 완료: 버전 {version}")
    
    # 모델 목록 확인
    logger.info("저장된 모델 목록:")
    models = deployer.list_models()
    for model in models:
        logger.info(f"- {model['name']} (v{model['version']})")
    
    # 예측 테스트
    logger.info("예측 테스트 중...")
    test_data = data[-10:].tolist()  # 최근 10개 데이터 포인트
    result = deployer.predict(test_data)
    logger.info(f"예측 결과: {result}")
    
    # API 서버 시작
    logger.info("API 서버 시작 중...")
    try:
        deployer.start()
    except KeyboardInterrupt:
        logger.info("서버 중지 중...")
        deployer.stop()
        logger.info("서버 중지 완료")

if __name__ == "__main__":
    main() 