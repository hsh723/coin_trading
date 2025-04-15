import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging
from typing import Dict, Any, List, Tuple

# 상위 디렉토리 임포트를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.ml.bayesian.ensemble_model import BayesianEnsembleModel
from src.data.collectors.binance import BinanceDataCollector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_crypto_data(symbol: str = "BTCUSDT", 
                    timeframe: str = "1d", 
                    limit: int = 180) -> pd.DataFrame:
    """
    바이낸스에서 암호화폐 데이터 가져오기
    
    Args:
        symbol: 심볼 (예: 'BTCUSDT')
        timeframe: 시간 프레임 (예: '1d', '4h', '1h')
        limit: 가져올 데이터 포인트 수
    
    Returns:
        DataFrame: 암호화폐 가격 데이터
    """
    try:
        # 바이낸스 데이터 수집기 초기화
        collector = BinanceDataCollector()
        
        # 데이터 가져오기
        data = collector.fetch_historical_ohlcv(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        
        # 시간 인덱스 설정
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        
        return data
    except Exception as e:
        logger.error(f"데이터 가져오기 실패: {str(e)}")
        
        # 예외 처리를 위한 더미 데이터 생성
        logger.warning("더미 데이터를 생성합니다.")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='D')
        
        # 랜덤한 가격 생성 (BTC 가격 범위 내에서)
        np.random.seed(42)  # 재현성을 위한 시드 설정
        close_prices = np.random.normal(30000, 5000, size=limit).cumsum()
        close_prices = np.abs(close_prices) + 20000  # 양수값으로 변환하고 오프셋 추가
        
        # 더미 DataFrame 생성
        data = pd.DataFrame({
            'open': close_prices * np.random.uniform(0.98, 1.0, size=limit),
            'high': close_prices * np.random.uniform(1.0, 1.05, size=limit),
            'low': close_prices * np.random.uniform(0.95, 1.0, size=limit),
            'close': close_prices,
            'volume': np.random.uniform(1000, 5000, size=limit) * 1000
        }, index=dates)
        
        return data

def create_ensemble_config() -> List[Dict[str, Any]]:
    """
    앙상블 모델 구성 생성
    
    Returns:
        모델 구성 리스트
    """
    return [
        {
            "type": "ar",
            "name": "AR(5) 모델",
            "params": {
                "ar_order": 5,
                "seasonality": True,
                "num_seasons": 7
            }
        },
        {
            "type": "gp",
            "name": "GP RBF 모델",
            "params": {
                "kernel_type": "rbf",
                "seasonality": True,
                "period": 7
            }
        },
        {
            "type": "gp",
            "name": "GP Matern 모델",
            "params": {
                "kernel_type": "matern52",
                "seasonality": False,
                "trend": True
            }
        },
        {
            "type": "structural",
            "name": "구조적 시계열 모델",
            "params": {
                "level": True,
                "trend": True,
                "seasonality": True,
                "season_period": 7,
                "damped_trend": True
            }
        }
    ]

def run_ensemble_prediction(symbol: str = "BTCUSDT",
                         n_forecast: int = 14,
                         ensemble_method: str = "weighted",
                         show_individual: bool = True) -> Dict[str, Any]:
    """
    앙상블 예측 실행 함수
    
    Args:
        symbol: 암호화폐 심볼
        n_forecast: 예측 일수
        ensemble_method: 앙상블 방법 ('mean', 'weighted', 'median', 'bayes')
        show_individual: 개별 모델 예측을 그래프에 표시할지 여부
        
    Returns:
        결과 딕셔너리 (예측값, 메트릭스 등)
    """
    # 데이터 가져오기
    logger.info(f"{symbol} 데이터를 가져오는 중...")
    data = fetch_crypto_data(symbol=symbol)
    
    # 종가 추출
    prices = data['close']
    
    # 학습 및 테스트 분할
    train_size = int(len(prices) * 0.8)
    train_data = prices[:train_size]
    test_data = prices[train_size:]
    
    logger.info(f"학습 데이터: {len(train_data)}개, 테스트 데이터: {len(test_data)}개")
    
    # 앙상블 모델 설정
    models_config = create_ensemble_config()
    
    # 앙상블 모델 생성
    logger.info(f"앙상블 모델 생성 중 (방식: {ensemble_method})...")
    ensemble = BayesianEnsembleModel(
        models_config=models_config,
        ensemble_method=ensemble_method
    )
    
    # MCMC 파라미터 설정 (학습 속도를 위해 간소화)
    sampling_params = {
        'draws': 300,
        'tune': 300,
        'chains': 2,
        'target_accept': 0.9
    }
    
    # 모델 학습
    logger.info("앙상블 모델 학습 중...")
    ensemble.fit(train_data, sampling_params=sampling_params)
    
    # 테스트 데이터 예측 및 성능 평가
    logger.info("테스트 데이터에 대한 예측 수행 중...")
    metrics = ensemble.evaluate(test_data)
    logger.info(f"앙상블 모델 테스트 성능: RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}")
    
    # 미래 예측
    logger.info(f"향후 {n_forecast}일 예측 중...")
    forecast, lower, upper = ensemble.predict(
        n_forecast=n_forecast, 
        adaptive_weights=True,
        test_data=test_data
    )
    
    # 각 모델별 가중치 출력 (가중 평균 방식인 경우)
    if ensemble_method == "weighted" and hasattr(ensemble, 'weights'):
        logger.info("모델 가중치:")
        for name, weight in zip(ensemble.model_names, ensemble.weights):
            logger.info(f"  - {name}: {weight:.4f}")
    
    # 결과 시각화
    fig = ensemble.plot_forecast(
        original_data=prices,
        forecast=forecast,
        lower=lower,
        upper=upper,
        title=f"{symbol} 가격 앙상블 예측 ({ensemble_method} 방식)",
        show_individual=show_individual
    )
    
    # 이미지 저장
    plt.savefig(f"{symbol}_ensemble_{ensemble_method}_prediction.png")
    plt.show()
    
    # 예측 결과 데이터프레임 생성
    # 미래 날짜 생성
    future_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=n_forecast, freq='D')
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': forecast,
        'lower_95': lower,
        'upper_95': upper
    })
    forecast_df.set_index('date', inplace=True)
    print(forecast_df)
    
    # 결과 반환
    return {
        'forecast': forecast,
        'lower': lower,
        'upper': upper,
        'metrics': metrics,
        'dates': future_dates
    }

def compare_ensemble_methods(symbol: str = "BTCUSDT", n_forecast: int = 14) -> None:
    """
    다양한 앙상블 방법 비교
    
    Args:
        symbol: 암호화폐 심볼
        n_forecast: 예측 일수
    """
    methods = ["mean", "weighted", "median", "bayes"]
    results = {}
    metrics = {}
    
    # 데이터 가져오기
    logger.info(f"{symbol} 데이터를 가져오는 중...")
    data = fetch_crypto_data(symbol=symbol)
    prices = data['close']
    
    # 학습 및 테스트 분할
    train_size = int(len(prices) * 0.8)
    train_data = prices[:train_size]
    test_data = prices[train_size:]
    
    # 각 앙상블 방법 실행
    for method in methods:
        logger.info(f"\n{method.upper()} 앙상블 방법 실행 중...")
        result = run_ensemble_prediction(
            symbol=symbol,
            n_forecast=n_forecast,
            ensemble_method=method,
            show_individual=False
        )
        
        results[method] = result
        metrics[method] = result['metrics']
    
    # 성능 비교 시각화
    plt.figure(figsize=(12, 8))
    
    # RMSE 비교
    plt.subplot(2, 1, 1)
    rmse_values = [m['rmse'] for m in metrics.values()]
    plt.bar(methods, rmse_values, color='skyblue')
    plt.title(f"{symbol} 앙상블 방법 RMSE 비교")
    plt.ylabel('RMSE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # MAE 비교
    plt.subplot(2, 1, 2)
    mae_values = [m['mae'] for m in metrics.values()]
    plt.bar(methods, mae_values, color='salmon')
    plt.title(f"{symbol} 앙상블 방법 MAE 비교")
    plt.ylabel('MAE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_ensemble_methods_comparison.png")
    plt.show()
    
    # 예측 비교 시각화
    plt.figure(figsize=(14, 7))
    
    # 원본 데이터 플롯
    plt.plot(prices.index, prices.values, label='실제 데이터', color='blue')
    
    # 미래 날짜 생성
    future_dates = results['mean']['dates']
    
    # 각 방법별 예측 플롯
    colors = ['red', 'green', 'purple', 'orange']
    for i, method in enumerate(methods):
        forecast = results[method]['forecast']
        lower = results[method]['lower']
        upper = results[method]['upper']
        
        plt.plot(future_dates, forecast, label=f'{method} 앙상블', color=colors[i])
        plt.fill_between(future_dates, lower, upper, alpha=0.1, color=colors[i])
    
    plt.title(f"{symbol} 가격 - 앙상블 방법 예측 비교")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_ensemble_forecasts_comparison.png")
    plt.show()
    
    # 결과 요약
    print("\n앙상블 방법 성능 비교:")
    for method in methods:
        print(f"{method.upper()}: RMSE = {metrics[method]['rmse']:.4f}, MAE = {metrics[method]['mae']:.4f}")

if __name__ == "__main__":
    # 앙상블 예측 실행
    run_ensemble_prediction(
        symbol="BTCUSDT",
        n_forecast=30,
        ensemble_method="weighted",
        show_individual=True
    )
    
    # 앙상블 방법 비교
    # compare_ensemble_methods(symbol="BTCUSDT", n_forecast=30) 