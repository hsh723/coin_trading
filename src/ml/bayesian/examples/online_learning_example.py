import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta

# 상위 디렉토리 임포트를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.ml.bayesian.online_learner import OnlineBayesianLearner
from src.data.collectors.binance import BinanceDataCollector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_historical_data(symbol: str = "BTCUSDT", 
                        timeframe: str = "1h", 
                        days: int = 30) -> pd.DataFrame:
    """
    과거 데이터 가져오기
    
    Args:
        symbol: 암호화폐 심볼
        timeframe: 시간 프레임
        days: 가져올 일수
        
    Returns:
        과거 데이터
    """
    try:
        # 바이낸스 데이터 수집기 초기화
        collector = BinanceDataCollector()
        
        # 데이터 가져오기
        limit = int((days * 24) / int(timeframe[:-1])) if 'h' in timeframe else days
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
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq=timeframe)
        
        # 랜덤한 가격 생성 (BTC 가격 범위 내에서)
        np.random.seed(42)  # 재현성을 위한 시드 설정
        close_prices = np.random.normal(30000, 5000, size=len(dates)).cumsum()
        close_prices = np.abs(close_prices) + 20000  # 양수값으로 변환하고 오프셋 추가
        
        # 더미 DataFrame 생성
        data = pd.DataFrame({
            'open': close_prices * np.random.uniform(0.98, 1.0, size=len(dates)),
            'high': close_prices * np.random.uniform(1.0, 1.05, size=len(dates)),
            'low': close_prices * np.random.uniform(0.95, 1.0, size=len(dates)),
            'close': close_prices,
            'volume': np.random.uniform(1000, 5000, size=len(dates)) * 1000
        }, index=dates)
        
        return data

def simulate_real_time_data(historical_data: pd.DataFrame, 
                          chunk_size: int = 7, 
                          delay: float = 0.5) -> pd.DataFrame:
    """
    실시간 데이터 시뮬레이션
    
    Args:
        historical_data: 과거 데이터
        chunk_size: 데이터 청크 크기
        delay: 청크 간 지연 시간 (초)
        
    Returns:
        실시간 데이터 (청크 단위로 반환)
    """
    for i in range(0, len(historical_data), chunk_size):
        end_idx = min(i + chunk_size, len(historical_data))
        
        # 현재 데이터 청크
        data_chunk = historical_data.iloc[i:end_idx]
        
        yield data_chunk
        
        if i + chunk_size < len(historical_data):
            time.sleep(delay)

def create_ensemble_config() -> List[Dict[str, Any]]:
    """
    앙상블 모델 구성 생성
    
    Returns:
        모델 구성 리스트
    """
    return [
        {
            "type": "ar",
            "name": "AR(3) 모델",
            "params": {
                "ar_order": 3,
                "seasonality": True,
                "num_seasons": 24  # 시간 단위 데이터의 경우 24시간 주기
            }
        },
        {
            "type": "gp",
            "name": "GP RBF 모델",
            "params": {
                "kernel_type": "rbf",
                "seasonality": True,
                "period": 24
            }
        },
        {
            "type": "structural",
            "name": "구조적 시계열 모델",
            "params": {
                "level": True,
                "trend": True,
                "seasonality": True,
                "season_period": 24,
                "damped_trend": True
            }
        }
    ]

def run_online_learning_example(symbol: str = "BTCUSDT",
                             timeframe: str = "1h",
                             days: int = 14,
                             model_type: str = "ar",
                             window_size: int = 168,  # 7일 (시간 단위)
                             update_freq: int = 24,  # 1일마다 업데이트
                             n_forecast: int = 48,   # 2일 예측
                             use_ensemble: bool = False) -> None:
    """
    온라인 학습 예제 실행
    
    Args:
        symbol: 암호화폐 심볼
        timeframe: 시간 프레임
        days: 시뮬레이션할 일수
        model_type: 모델 유형
        window_size: 학습 윈도우 크기
        update_freq: 업데이트 주기
        n_forecast: 예측 기간
        use_ensemble: 앙상블 모델 사용 여부
    """
    # 과거 데이터 가져오기
    logger.info(f"{symbol} {timeframe} 데이터 가져오는 중...")
    data = fetch_historical_data(symbol=symbol, timeframe=timeframe, days=days)
    
    # 종가 추출
    prices = data['close']
    
    # 모델 파라미터 설정
    if model_type == "ar":
        model_params = {
            "ar_order": 5,
            "seasonality": True,
            "num_seasons": 24 if timeframe == "1h" else 7
        }
    elif model_type == "gp":
        model_params = {
            "kernel_type": "matern52",
            "seasonality": True,
            "period": 24 if timeframe == "1h" else 7
        }
    elif model_type == "structural":
        model_params = {
            "level": True,
            "trend": True,
            "seasonality": True,
            "season_period": 24 if timeframe == "1h" else 7,
            "damped_trend": True
        }
    else:
        model_params = {}
    
    # 온라인 학습기 초기화
    if use_ensemble:
        logger.info("앙상블 온라인 학습기 초기화 중...")
        ensemble_config = create_ensemble_config()
        learner = OnlineBayesianLearner(
            model_type="ensemble",
            window_size=window_size,
            update_freq=update_freq,
            save_dir=f"./models/{symbol}",
            ensemble_config=ensemble_config,
            ensemble_method="weighted"
        )
    else:
        logger.info(f"{model_type.upper()} 온라인 학습기 초기화 중...")
        learner = OnlineBayesianLearner(
            model_type=model_type,
            model_params=model_params,
            window_size=window_size,
            update_freq=update_freq,
            save_dir=f"./models/{symbol}"
        )
    
    # 학습 및 예측 결과 저장 변수
    predictions = []
    actual_values = []
    prediction_dates = []
    update_points = []
    
    # 실시간 데이터 시뮬레이션
    logger.info("실시간 데이터 스트림 시뮬레이션 시작...")
    
    # 초기 데이터 점수
    initial_data_count = min(60, len(prices) // 3)  # 전체 데이터의 1/3, 최대 60개 포인트
    logger.info(f"초기 {initial_data_count}개 데이터 포인트로 모델 초기화...")
    
    # 초기 데이터 추가
    initial_data = prices.iloc[:initial_data_count]
    learner.add_data(initial_data, update_model=True)
    
    # 남은 데이터로 온라인 학습 시뮬레이션
    remaining_data = prices.iloc[initial_data_count:]
    
    total_updates = 0
    for chunk in simulate_real_time_data(remaining_data, chunk_size=update_freq, delay=0.2):
        # 데이터 추가 및 모델 업데이트
        updated = learner.add_data(chunk, update_model=True)
        
        if updated:
            total_updates += 1
            update_points.append(len(predictions))
            
            # 현재 시점 예측 (n_forecast 기간 동안)
            forecast, lower, upper = learner.predict(n_forecast=n_forecast)
            
            # 예측 시점의 실제 값 (가능한 범위까지)
            last_date = chunk.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=n_forecast, freq=timeframe)
            
            actual = []
            for date in future_dates:
                if date in prices.index:
                    actual.append(prices.loc[date])
                else:
                    actual.append(np.nan)
            
            # 예측 결과 저장
            predictions.append((forecast, lower, upper))
            actual_values.append(actual)
            prediction_dates.append(future_dates)
            
            # 현재 예측 시각화
            plt.figure(figsize=(10, 6))
            
            # 과거 데이터
            history_data = pd.Series([t[1] for t in learner.data_buffer], index=[t[0] for t in learner.data_buffer])
            plt.plot(history_data.index, history_data.values, label='과거 데이터', color='blue')
            
            # 예측
            plt.plot(future_dates, forecast, label='예측', color='red')
            plt.fill_between(future_dates, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
            
            # 실제 값 (아직 알 수 있는 범위까지)
            valid_indices = ~np.isnan(actual)
            if np.any(valid_indices):
                valid_dates = future_dates[valid_indices]
                valid_actual = np.array(actual)[valid_indices]
                plt.plot(valid_dates, valid_actual, 'g--', label='실제 미래 값', marker='o')
            
            plt.title(f"{symbol} {timeframe} 예측 (업데이트 #{total_updates})")
            plt.xlabel('시간')
            plt.ylabel('가격')
            plt.legend()
            plt.grid(True)
            
            # 이미지 저장
            if not os.path.exists('predictions'):
                os.makedirs('predictions')
            plt.savefig(f"predictions/{symbol}_{model_type}_update_{total_updates}.png")
            plt.close()
            
            logger.info(f"업데이트 #{total_updates} 완료, 예측 저장됨")
        
        else:
            logger.info("모델 업데이트 기준 미달, 업데이트 건너뜀")
    
    # 최종 성능 평가
    if predictions:
        # 모든 예측에 대한 평가
        eval_results = []
        
        for i, (pred, actual, dates) in enumerate(zip(predictions, actual_values, prediction_dates)):
            forecast = pred[0]  # 예측값 (평균)
            
            # 실제 값 있는 인덱스만 필터링
            valid_indices = ~np.isnan(actual)
            if np.sum(valid_indices) > 0:
                y_pred = forecast[valid_indices]
                y_true = np.array(actual)[valid_indices]
                
                # 평가 지표 계산
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                mae = np.mean(np.abs(y_true - y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                eval_results.append({
                    'update': i + 1,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'horizon': np.sum(valid_indices)
                })
        
        # 결과 요약
        if eval_results:
            df_results = pd.DataFrame(eval_results)
            print("\n예측 성능 요약:")
            print(df_results)
            
            # 성능 추이 시각화
            plt.figure(figsize=(12, 8))
            
            plt.subplot(3, 1, 1)
            plt.plot(df_results['update'], df_results['rmse'], marker='o')
            plt.title('RMSE 추이')
            plt.grid(True)
            
            plt.subplot(3, 1, 2)
            plt.plot(df_results['update'], df_results['mae'], marker='o', color='orange')
            plt.title('MAE 추이')
            plt.grid(True)
            
            plt.subplot(3, 1, 3)
            plt.plot(df_results['update'], df_results['mape'], marker='o', color='green')
            plt.title('MAPE 추이')
            plt.xlabel('업데이트 횟수')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"predictions/{symbol}_{model_type}_performance.png")
            plt.show()
    
    # 성능 추이 시각화 (클래스 내부 기록 기반)
    try:
        if learner.performance_history:
            perf_fig = learner.plot_performance()
            plt.savefig(f"predictions/{symbol}_{model_type}_internal_performance.png")
            plt.show()
    except Exception as e:
        logger.warning(f"성능 추이 시각화 실패: {str(e)}")
    
    # 최종 예측 시각화
    final_fig = learner.plot_forecast(n_forecast=n_forecast*2)
    plt.savefig(f"predictions/{symbol}_{model_type}_final_forecast.png")
    plt.show()

def compare_models_online(symbol: str = "BTCUSDT",
                       timeframe: str = "1h",
                       days: int = 14) -> None:
    """
    다양한 모델의 온라인 학습 성능 비교
    
    Args:
        symbol: 암호화폐 심볼
        timeframe: 시간 프레임
        days: 시뮬레이션할 일수
    """
    models = ["ar", "gp", "structural", "ensemble"]
    model_names = {
        "ar": "자기회귀",
        "gp": "가우시안 프로세스",
        "structural": "구조적 시계열",
        "ensemble": "앙상블"
    }
    
    for model in models:
        logger.info(f"\n{'='*50}")
        logger.info(f"{model_names[model]} 모델 온라인 학습 실행")
        logger.info(f"{'='*50}")
        
        use_ensemble = model == "ensemble"
        
        run_online_learning_example(
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            model_type=model if not use_ensemble else "ar",
            use_ensemble=use_ensemble
        )
        
        logger.info(f"{model_names[model]} 모델 실행 완료")
        logger.info(f"{'='*50}\n")

if __name__ == "__main__":
    # 단일 모델 온라인 학습 예제
    run_online_learning_example(
        symbol="BTCUSDT",
        timeframe="1h",
        days=7,
        model_type="ar"
    )
    
    # 다양한 모델 비교
    # compare_models_online(
    #     symbol="BTCUSDT",
    #     timeframe="1h",
    #     days=7
    # ) 