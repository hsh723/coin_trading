import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import pymc3 as pm
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 경로 추가하여 모듈 import 가능하게 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.bayesian.model_factory import BayesianModelFactory
from src.ml.bayesian.ensemble_model import BayesianEnsembleModel
from src.ml.bayesian.online_learner import OnlineBayesianLearner
from src.ml.bayesian.model_visualization import BayesianModelVisualizer

def load_crypto_data(crypto='BTC', timeframe='1d', n_samples=180):
    """샘플 암호화폐 데이터 로드 함수 (샘플 데이터 또는 API에서 로드)"""
    try:
        # 실제 구현에서는 데이터 소스에서 데이터를 가져오는 코드
        # 이 예제에서는 가상의 데이터 생성
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq=timeframe)
        
        # 랜덤 시드 설정 (재현성을 위해)
        np.random.seed(42)
        
        # 랜덤 워크로 가격 데이터 생성
        base_price = 30000  # BTC 기준가
        if crypto == 'ETH':
            base_price = 2000
        elif crypto == 'XRP':
            base_price = 0.5
        
        # 추세, 계절성, 및 노이즈가 있는 시계열 생성
        t = np.arange(n_samples)
        trend = 0.1 * t
        seasonality = 10 * np.sin(2 * np.pi * t / 30)  # 30일 주기의 계절성
        noise = np.random.normal(0, 5, n_samples)
        
        prices = base_price + trend + seasonality + np.cumsum(noise) + np.random.normal(0, 50, n_samples)
        
        return pd.Series(prices, index=dates, name=f'{crypto}-USD')
    
    except Exception as e:
        logger.error(f"데이터 로드 오류: {str(e)}")
        # 오류 발생 시 간단한 랜덤 데이터 생성
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq=timeframe)
        prices = np.random.normal(1000, 100, n_samples)
        return pd.Series(prices, index=dates, name=f'{crypto}-USD')

def main():
    """모델 시각화 예제 메인 함수"""
    logger.info("베이지안 모델 시각화 예제 시작")
    
    # 시각화 클래스 인스턴스 생성
    visualizer = BayesianModelVisualizer(save_dir='./plots/bayesian_models')
    
    # 데이터 로드
    crypto_data = load_crypto_data(crypto='BTC', n_samples=120)
    logger.info(f"데이터 로드 완료: {len(crypto_data)} 데이터 포인트")
    
    # 학습/테스트 분할
    train_data = crypto_data[:-10]
    test_data = crypto_data[-10:]
    
    # 1. 기본 AR 모델 예측 시각화
    logger.info("AR 모델 학습 및 예측 시작")
    ar_model = BayesianModelFactory.get_model("ar", ar_order=3, seasonality=True, num_seasons=7)
    
    # 간소화된 샘플링 파라미터 (빠른 실행을 위해)
    sampling_params = {
        'draws': 500,
        'tune': 300,
        'chains': 2,
    }
    
    ar_model.fit(train_data, sampling_params=sampling_params)
    forecast, lower, upper = ar_model.predict(n_forecast=15)
    
    # 예측 결과 시각화
    logger.info("AR 모델 예측 시각화")
    fig = visualizer.plot_forecast(
        original_data=crypto_data, 
        forecast=forecast, 
        lower=lower, 
        upper=upper,
        title="비트코인 가격 예측",
        y_label="가격 (USD)",
        model_name="AR(3) + 계절성",
        save_path="ar_model_forecast.png"
    )
    plt.close(fig)
    
    # 사후 분포 시각화
    logger.info("AR 모델 사후 분포 시각화")
    fig = visualizer.plot_trace(
        ar_model.trace, 
        title="AR 모델 사후 분포",
        save_path="ar_model_trace.png"
    )
    plt.close(fig)
    
    # 2. 여러 모델 비교 시각화
    logger.info("다중 모델 학습 시작")
    
    # 가우시안 프로세스 모델 학습
    gp_model = BayesianModelFactory.get_model("gp", kernel_type="rbf", seasonality=True, period=7)
    gp_model.fit(train_data, sampling_params=sampling_params)
    gp_forecast, gp_lower, gp_upper = gp_model.predict(n_forecast=15)
    
    # 구조적 시계열 모델 학습
    struct_model = BayesianModelFactory.get_model("structural", trend=True, seasonality=True, season_period=7)
    struct_model.fit(train_data, sampling_params=sampling_params)
    struct_forecast, struct_lower, struct_upper = struct_model.predict(n_forecast=15)
    
    # 모델별 예측 결과 및 성능 지표
    model_forecasts = {
        "AR(3)": (forecast, lower, upper),
        "가우시안 프로세스": (gp_forecast, gp_lower, gp_upper),
        "구조적 시계열": (struct_forecast, struct_lower, struct_upper)
    }
    
    # 테스트 데이터로 성능 평가
    metrics = {}
    for name, (forecast_vals, _, _) in model_forecasts.items():
        # 테스트 기간 예측 재수행
        if name == "AR(3)":
            test_forecast, _, _ = ar_model.predict(n_forecast=len(test_data))
        elif name == "가우시안 프로세스":
            test_forecast, _, _ = gp_model.predict(n_forecast=len(test_data))
        elif name == "구조적 시계열":
            test_forecast, _, _ = struct_model.predict(n_forecast=len(test_data))
        
        # 성능 지표 계산
        rmse = np.sqrt(np.mean((test_data.values - test_forecast) ** 2))
        mae = np.mean(np.abs(test_data.values - test_forecast))
        
        metrics[name] = {
            "rmse": rmse,
            "mae": mae
        }
    
    # 모델 비교 시각화
    logger.info("다중 모델 비교 시각화")
    fig = visualizer.plot_multi_model_comparison(
        original_data=crypto_data,
        model_forecasts=model_forecasts,
        title="비트코인 가격 예측 모델 비교",
        y_label="가격 (USD)",
        metrics=metrics,
        show_uncertainty=True,
        save_path="model_comparison.png"
    )
    plt.close(fig)
    
    # 성능 지표 시각화
    logger.info("모델 성능 지표 시각화")
    fig = visualizer.plot_performance_metrics(
        metrics=metrics,
        title="비트코인 예측 모델 성능 비교",
        primary_metric="rmse",
        save_path="model_metrics.png"
    )
    plt.close(fig)
    
    # 3. 앙상블 모델 시각화
    logger.info("앙상블 모델 학습 시작")
    
    # 앙상블 모델 구성
    models_config = [
        {"model_type": "ar", "ar_order": 3, "seasonality": True, "num_seasons": 7},
        {"model_type": "ar", "ar_order": 5, "seasonality": False},
        {"model_type": "gp", "kernel_type": "rbf", "seasonality": True, "period": 7},
        {"model_type": "structural", "trend": True, "seasonality": True, "season_period": 7}
    ]
    
    # 앙상블 모델 학습
    ensemble_model = BayesianEnsembleModel(models_config, ensemble_method="weighted")
    ensemble_model.fit(train_data, sampling_params=sampling_params)
    
    # 앙상블 예측
    ensemble_forecast, ensemble_lower, ensemble_upper = ensemble_model.predict(n_forecast=15)
    
    # 앙상블 모델 시각화
    logger.info("앙상블 모델 예측 시각화")
    fig = visualizer.plot_forecast(
        original_data=crypto_data,
        forecast=ensemble_forecast,
        lower=ensemble_lower,
        upper=ensemble_upper,
        title="비트코인 가격 예측",
        y_label="가격 (USD)",
        model_name="베이지안 앙상블",
        save_path="ensemble_forecast.png"
    )
    plt.close(fig)
    
    # 4. 온라인 학습 성능 추적 시각화
    logger.info("온라인 학습 시뮬레이션 시작")
    
    # 더 긴 데이터셋 생성 (온라인 학습용)
    online_data = load_crypto_data(crypto='BTC', n_samples=180)
    
    # 온라인 학습기 초기화
    online_learner = OnlineBayesianLearner(
        model_type="ar",
        model_params={"ar_order": 3, "seasonality": True, "num_seasons": 7},
        window_size=60,
        update_freq=5
    )
    
    # 온라인 학습 시뮬레이션
    performance_history = []
    
    # 초기 학습
    initial_data = online_data[:60]
    online_learner.initialize(initial_data)
    
    # 점진적 업데이트 및 평가
    for i in range(60, len(online_data), 1):
        # 새 데이터 추가
        new_point = online_data.iloc[i:i+1]
        online_learner.update(new_point)
        
        # 예측 및 평가 (5포인트 예측)
        if i + 5 <= len(online_data):
            forecast, lower, upper = online_learner.predict(n_forecast=5)
            actual = online_data.iloc[i:i+5].values
            
            # 성능 평가
            rmse = np.sqrt(np.mean((actual - forecast[:len(actual)]) ** 2))
            
            # 기록 추가
            is_update = online_learner.update_count > 0 and (online_learner.update_count * online_learner.update_freq) % i == 0
            performance_history.append({
                "timestamp": online_data.index[i],
                "metrics": {"rmse": rmse},
                "is_update": is_update
            })
    
    # 온라인 학습 성능 시각화
    logger.info("온라인 학습 성능 추적 시각화")
    fig = visualizer.plot_online_learning_performance(
        performance_history=performance_history,
        metric="rmse",
        title="비트코인 예측 온라인 학습 성능 추적",
        save_path="online_learning_performance.png"
    )
    plt.close(fig)
    
    logger.info("베이지안 모델 시각화 예제 완료")
    print("모든 시각화가 './plots/bayesian_models' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 