import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 상위 디렉토리 임포트를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.ml.bayesian.model_factory import BayesianModelFactory
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
        
        # 랜덤한 가격 생성
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

def evaluate_forecast(actual: np.ndarray, 
                    forecast: np.ndarray) -> Dict[str, float]:
    """
    예측 성능 평가 지표 계산
    
    Args:
        actual: 실제 값
        forecast: 예측 값
        
    Returns:
        Dict: 성능 지표
    """
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    r2 = r2_score(actual, forecast)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def compare_models(models_config: List[Dict], 
                 symbol: str = "BTCUSDT",
                 n_forecast: int = 14) -> Tuple[pd.DataFrame, Dict]:
    """
    여러 베이지안 시계열 모델 비교
    
    Args:
        models_config: 모델 설정 목록
        symbol: 암호화폐 심볼
        n_forecast: 예측 일수
        
    Returns:
        성능 지표 DataFrame 및 각 모델의 예측 결과
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
    
    # MCMC 파라미터 설정 (모든 모델에 공통 적용)
    sampling_params = {
        'draws': 500,
        'tune': 500,
        'chains': 2,
        'target_accept': 0.95
    }
    
    # 모델별 학습 및 예측
    results = {}
    metrics_list = []
    
    for config in models_config:
        model_type = config['type']
        model_name = config['name']
        model_params = config.get('params', {})
        
        logger.info(f"{model_name} 모델 생성 및 학습 중...")
        
        # 모델 생성
        model = BayesianModelFactory.get_model(model_type, **model_params)
        
        # 모델 학습
        model.fit(train_data, sampling_params=sampling_params)
        
        # 테스트 데이터 예측
        forecast_test, lower_test, upper_test = model.predict(n_forecast=len(test_data))
        
        # 성능 평가
        metrics = evaluate_forecast(test_data.values, forecast_test)
        metrics['model'] = model_name
        metrics_list.append(metrics)
        
        # 미래 예측
        forecast, lower, upper = model.predict(n_forecast=n_forecast)
        
        # 미래 날짜 생성
        future_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=n_forecast, freq='D')
        
        # 결과 저장
        results[model_name] = {
            'model': model,
            'test_forecast': forecast_test,
            'test_lower': lower_test,
            'test_upper': upper_test,
            'future_forecast': forecast,
            'future_lower': lower,
            'future_upper': upper,
            'future_dates': future_dates,
            'metrics': metrics
        }
    
    # 성능 지표 DataFrame 생성
    metrics_df = pd.DataFrame(metrics_list).set_index('model')
    
    return metrics_df, results

def plot_model_comparison(metrics_df: pd.DataFrame, results: Dict, 
                       train_data: pd.Series, test_data: pd.Series, 
                       symbol: str) -> None:
    """
    모델 비교 시각화
    
    Args:
        metrics_df: 성능 지표 DataFrame
        results: 모델별 예측 결과
        train_data: 학습 데이터
        test_data: 테스트 데이터
        symbol: 암호화폐 심볼
    """
    # 모델 수
    n_models = len(results)
    
    # 1. 성능 지표 시각화
    plt.figure(figsize=(12, 6))
    ax = metrics_df[['rmse', 'mae']].plot(kind='bar', figsize=(12, 6))
    ax.set_title(f"{symbol} 예측 모델 성능 비교")
    ax.set_ylabel('오차')
    ax.set_xticklabels(metrics_df.index, rotation=45)
    
    # 값 표시
    for i, v in enumerate(metrics_df['rmse']):
        ax.text(i-0.1, v + 50, f"{v:.1f}", ha='center')
    
    for i, v in enumerate(metrics_df['mae']):
        ax.text(i+0.1, v + 50, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_model_metrics_comparison.png")
    plt.show()
    
    # 2. 예측 결과 시각화
    plt.figure(figsize=(14, 10))
    
    # 테스트 세트 예측
    plt.subplot(2, 1, 1)
    plt.plot(train_data.index, train_data.values, label='학습 데이터', color='blue')
    plt.plot(test_data.index, test_data.values, label='실제 테스트 데이터', color='black', linestyle='--')
    
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    
    for i, (model_name, model_result) in enumerate(results.items()):
        plt.plot(test_data.index, model_result['test_forecast'], label=f'{model_name} 예측', 
                color=colors[i % len(colors)])
    
    plt.title(f"{symbol} 테스트 데이터 예측 비교")
    plt.legend()
    plt.grid(True)
    
    # 미래 예측
    plt.subplot(2, 1, 2)
    all_data = pd.concat([train_data, test_data])
    plt.plot(all_data.index, all_data.values, label='실제 데이터', color='blue')
    
    for i, (model_name, model_result) in enumerate(results.items()):
        plt.plot(model_result['future_dates'], model_result['future_forecast'], 
                label=f'{model_name} 예측', color=colors[i % len(colors)])
        plt.fill_between(model_result['future_dates'], 
                         model_result['future_lower'], 
                         model_result['future_upper'], 
                         color=colors[i % len(colors)], 
                         alpha=0.2)
    
    plt.title(f"{symbol} 향후 예측 비교")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_model_forecast_comparison.png")
    plt.show()
    
    # 3. 모델별 예측값 테이블
    future_dates = list(results.values())[0]['future_dates']
    forecasts = {}
    
    for model_name, model_result in results.items():
        forecasts[f"{model_name}"] = model_result['future_forecast']
        forecasts[f"{model_name}_lower"] = model_result['future_lower']
        forecasts[f"{model_name}_upper"] = model_result['future_upper']
    
    forecast_df = pd.DataFrame(forecasts, index=future_dates)
    logger.info("\n예측 결과:")
    print(forecast_df)

def run_model_comparison(symbol: str = "BTCUSDT", n_forecast: int = 14) -> None:
    """
    베이지안 시계열 모델 비교 실행
    
    Args:
        symbol: 암호화폐 심볼
        n_forecast: 예측 일수
    """
    # 모델 설정
    models_config = [
        {
            'type': 'ar',
            'name': 'AR(5)',
            'params': {
                'ar_order': 5,
                'seasonality': False
            }
        },
        {
            'type': 'ar',
            'name': 'AR(5) + 계절성',
            'params': {
                'ar_order': 5,
                'seasonality': True,
                'num_seasons': 7
            }
        },
        {
            'type': 'gp',
            'name': 'GP(RBF)',
            'params': {
                'kernel_type': 'rbf',
                'seasonality': False
            }
        },
        {
            'type': 'gp',
            'name': 'GP(Matern) + 계절성',
            'params': {
                'kernel_type': 'matern52',
                'seasonality': True,
                'period': 7
            }
        },
        {
            'type': 'structural',
            'name': '구조적 시계열',
            'params': {
                'level': True,
                'trend': True,
                'seasonality': True,
                'season_period': 7,
                'damped_trend': True
            }
        }
    ]
    
    # 데이터 가져오기
    data = fetch_crypto_data(symbol=symbol)
    prices = data['close']
    
    # 학습 및 테스트 분할
    train_size = int(len(prices) * 0.8)
    train_data = prices[:train_size]
    test_data = prices[train_size:]
    
    # 모델 비교
    metrics_df, results = compare_models(models_config, symbol, n_forecast)
    
    # 비교 결과 시각화
    plot_model_comparison(metrics_df, results, train_data, test_data, symbol)
    
    # 결과 요약
    logger.info("\n성능 지표 요약:")
    print(metrics_df)
    
    # 최고 성능 모델 확인
    best_model = metrics_df['rmse'].idxmin()
    logger.info(f"\n최고 성능 모델 (RMSE 기준): {best_model}")
    logger.info(f"RMSE: {metrics_df.loc[best_model, 'rmse']:.2f}")
    logger.info(f"MAE: {metrics_df.loc[best_model, 'mae']:.2f}")
    logger.info(f"MAPE: {metrics_df.loc[best_model, 'mape']:.2f}%")

if __name__ == "__main__":
    # 모델 비교 실행
    run_model_comparison(
        symbol="BTCUSDT",
        n_forecast=30
    ) 