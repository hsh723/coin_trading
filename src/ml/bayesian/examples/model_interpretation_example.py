import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import pymc3 as pm
import logging
from scipy import stats

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 경로 추가하여 모듈 import 가능하게 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.bayesian.model_factory import BayesianModelFactory
from src.ml.bayesian.ensemble_model import BayesianEnsembleModel
from src.ml.bayesian.model_interpretation import BayesianModelInterpreter

def load_crypto_data(crypto='BTC', timeframe='1d', n_samples=180):
    """샘플 암호화폐 데이터 로드 함수 (가상의 데이터 생성)"""
    try:
        # 가상의 데이터 생성
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
    """모델 해석 예제 메인 함수"""
    logger.info("베이지안 모델 해석 예제 시작")
    
    # 해석 클래스 인스턴스 생성
    interpreter = BayesianModelInterpreter(save_dir='./interpretations/bayesian_models')
    
    # 데이터 로드
    crypto_data = load_crypto_data(crypto='BTC', n_samples=120)
    logger.info(f"데이터 로드 완료: {len(crypto_data)} 데이터 포인트")
    
    # 학습/테스트 분할
    train_data = crypto_data[:-20]
    test_data = crypto_data[-20:]
    
    # 1. AR 모델 학습 및 해석
    logger.info("AR 모델 학습 시작")
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
    
    # 예측 수행
    forecast, lower, upper = ar_model.predict(n_forecast=30)
    
    # 2. 모델 파라미터 요약 및 중요도 분석
    logger.info("모델 파라미터 요약 통계 계산")
    param_summary = interpreter.summarize_parameters(ar_model.trace)
    print("\n모델 파라미터 요약 통계:")
    print(param_summary)
    
    # 파라미터 중요도 시각화
    logger.info("파라미터 중요도 시각화")
    fig = interpreter.plot_parameter_importance(
        trace=ar_model.trace,
        title="AR 모델 파라미터 중요도",
        save_path="ar_parameter_importance.png"
    )
    plt.close(fig)
    
    # 3. 사후 예측 검사
    logger.info("사후 예측 검사 수행")
    # pymc3 모델 객체 필요
    ar_model_pymc = ar_model.model
    fig = interpreter.posterior_predictive_check(
        model=ar_model_pymc,
        trace=ar_model.trace,
        observed_data=train_data,
        title="AR 모델 사후 예측 검사",
        save_path="ar_posterior_predictive_check.png"
    )
    plt.close(fig)
    
    # 4. 불확실성 분석
    logger.info("예측 불확실성 분석")
    fig, uncertainty_metrics = interpreter.analyze_uncertainty(
        forecast=forecast,
        lower=lower,
        upper=upper,
        times=pd.date_range(start=crypto_data.index[-1], periods=len(forecast), freq='D'),
        title="AR 모델 예측 불확실성 분석",
        save_path="ar_uncertainty_analysis.png"
    )
    plt.close(fig)
    
    print("\n불확실성 지표:")
    for key, value in uncertainty_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 5. 파라미터 민감도 분석
    logger.info("파라미터 민감도 분석")
    
    # 모델 생성 함수 정의
    def create_ar_model(ar_order=3, seasonality=True, num_seasons=7):
        return BayesianModelFactory.get_model("ar", ar_order=ar_order, 
                                             seasonality=seasonality, 
                                             num_seasons=num_seasons)
    
    # 기본 파라미터
    baseline_params = {'ar_order': 3, 'seasonality': True, 'num_seasons': 7}
    
    # 민감도 분석을 수행할 파라미터 범위
    param_ranges = {
        'ar_order': [1, 2, 3, 4, 5],
        'num_seasons': [5, 7, 10, 14, 21]
    }
    
    # RMSE 계산 함수
    def rmse_metric(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred[:len(y_true)]) ** 2))
    
    # 민감도 분석 수행
    fig = interpreter.parameter_sensitivity(
        model_func=create_ar_model,
        baseline_params=baseline_params,
        param_ranges=param_ranges,
        metric_func=rmse_metric,
        data=test_data.values,
        title="AR 모델 파라미터 민감도 분석",
        save_path="ar_parameter_sensitivity.png"
    )
    plt.close(fig)
    
    # 6. 구조적 시계열 모델 학습 및 구성요소 분석
    logger.info("구조적 시계열 모델 학습 시작")
    structural_model = BayesianModelFactory.get_model(
        "structural", 
        level=True,
        trend=True,
        seasonality=True,
        season_period=7,
        damped_trend=False
    )
    
    # 모델 학습
    structural_model.fit(train_data, sampling_params=sampling_params)
    
    # 예측 수행
    struct_forecast, struct_lower, struct_upper = structural_model.predict(n_forecast=30)
    
    # 구성요소 분석 
    # 참고: 실제 구현에서는 구조적 모델이 구성요소 메서드를 제공해야 함
    try:
        logger.info("예측 구성요소 분석")
        fig = interpreter.analyze_forecast_components(
            model=structural_model,
            n_forecast=30,
            component_names=['level', 'trend', 'seasonality'],
            dates=pd.date_range(start=crypto_data.index[-1], periods=30, freq='D'),
            title="구조적 시계열 모델 구성요소 분석",
            save_path="structural_components.png"
        )
        plt.close(fig)
    except Exception as e:
        logger.warning(f"구성요소 분석 실패: {str(e)}")
    
    # 7. 해석 보고서 생성
    logger.info("모델 해석 보고서 생성")
    report_path = interpreter.generate_interpretation_report(
        model=ar_model,
        trace=ar_model.trace,
        observed_data=train_data,
        forecast=forecast,
        lower=lower,
        upper=upper,
        report_path="ar_model_interpretation_report.html"
    )
    
    logger.info(f"모델 해석 보고서가 생성되었습니다: {report_path}")
    logger.info("베이지안 모델 해석 예제 완료")
    print(f"\n모든 해석 결과는 '{interpreter.save_dir}' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 