import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 경로 추가하여 모듈 import 가능하게 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.bayesian.model_factory import BayesianModelFactory
from src.ml.bayesian.anomaly_detection import BayesianAnomalyDetector

def generate_sample_data(n_samples=200, anomaly_rate=0.05):
    """샘플 시계열 데이터 생성 (이상치 포함)"""
    # 기본 시계열 생성
    t = np.arange(n_samples)
    trend = 0.1 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, 2, n_samples)
    
    # 기본 시계열
    base_series = trend + seasonality + noise
    
    # 이상치 생성
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    anomalies = np.random.normal(0, 20, n_anomalies)
    
    # 이상치 추가
    series = base_series.copy()
    series[anomaly_indices] += anomalies
    
    # 시간 인덱스 생성
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_samples), 
                         periods=n_samples, freq='D')
    
    return pd.Series(series, index=dates)

def main():
    """이상치 탐지 예제 메인 함수"""
    logger.info("베이지안 이상치 탐지 예제 시작")
    
    # 1. 샘플 데이터 생성
    logger.info("샘플 데이터 생성")
    data = generate_sample_data(n_samples=200, anomaly_rate=0.05)
    
    # 2. 학습/테스트 분할
    train_data = data[:-50]
    test_data = data[-50:]
    
    # 3. AR 모델 학습
    logger.info("AR 모델 학습")
    ar_model = BayesianModelFactory.get_model("ar", ar_order=3, seasonality=True, num_seasons=7)
    
    # 간소화된 샘플링 파라미터 (빠른 실행을 위해)
    sampling_params = {
        'draws': 500,
        'tune': 300,
        'chains': 2,
        'return_inferencedata': True
    }
    
    # 모델 학습
    ar_model.fit(train_data, sampling_params=sampling_params)
    
    # 4. 테스트 데이터 예측
    logger.info("테스트 데이터 예측")
    forecast, lower, upper = ar_model.predict(n_forecast=len(test_data))
    
    # 5. 이상치 탐지기 초기화
    logger.info("이상치 탐지기 초기화")
    detector = BayesianAnomalyDetector(
        model=ar_model,
        threshold=0.95,
        window_size=10,
        min_anomaly_score=0.5,
        save_dir="./anomaly_detection"
    )
    
    # 6. 이상치 탐지
    logger.info("이상치 탐지 수행")
    detection_result = detector.detect_anomalies(
        observed=test_data,
        predicted=forecast,
        return_scores=True
    )
    
    # 7. 이상치 분석
    logger.info("이상치 분석 수행")
    analysis = detector.analyze_anomalies(
        observed=test_data,
        predicted=forecast,
        timestamps=test_data.index
    )
    
    # 8. 이상치 시각화
    logger.info("이상치 시각화")
    fig = detector.plot_anomalies(
        observed=test_data,
        predicted=forecast,
        timestamps=test_data.index,
        save_path="./anomaly_detection/anomalies.png"
    )
    
    # 9. 이상치 분석 보고서 생성
    logger.info("이상치 분석 보고서 생성")
    report = detector.generate_report(
        observed=test_data,
        predicted=forecast,
        timestamps=test_data.index,
        save_path="./anomaly_detection/anomaly_report.json"
    )
    
    # 10. 임계값 자동 조정
    logger.info("임계값 자동 조정")
    new_threshold = detector.update_threshold(
        observed=test_data,
        predicted=forecast,
        target_anomaly_rate=0.05
    )
    
    # 결과 출력
    logger.info("이상치 탐지 결과 요약:")
    logger.info(f"- 탐지된 이상치 수: {analysis['n_anomalies']}")
    logger.info(f"- 이상치 비율: {analysis['n_anomalies'] / len(test_data):.2%}")
    logger.info(f"- 평균 이상치 점수: {np.mean(analysis['anomaly_scores']):.4f}")
    logger.info(f"- 조정된 임계값: {new_threshold:.4f}")
    
    logger.info("베이지안 이상치 탐지 예제 완료")

if __name__ == "__main__":
    main() 