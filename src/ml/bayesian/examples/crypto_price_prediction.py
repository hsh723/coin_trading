import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging
from typing import Dict, Any

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

def run_crypto_price_prediction(model_type: str = "ar", 
                             symbol: str = "BTCUSDT",
                             n_forecast: int = 14,
                             model_params: Dict[str, Any] = None) -> None:
    """
    암호화폐 가격 예측 실행 함수
    
    Args:
        model_type: 사용할 모델 유형 ('ar', 'gp', 'structural')
        symbol: 암호화폐 심볼
        n_forecast: 예측 일수
        model_params: 모델 파라미터
    """
    # 기본 모델 파라미터
    if model_params is None:
        if model_type == "ar":
            model_params = {
                "ar_order": 5,
                "seasonality": True,
                "num_seasons": 7
            }
        elif model_type == "gp":
            model_params = {
                "kernel_type": "matern52",
                "seasonality": True,
                "period": 7
            }
        elif model_type == "structural":
            model_params = {
                "level": True,
                "trend": True,
                "seasonality": True,
                "season_period": 7,
                "damped_trend": True
            }
    
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
    
    # 모델 생성
    logger.info(f"{model_type} 모델 생성 중...")
    model = BayesianModelFactory.get_model(model_type, **model_params)
    
    # MCMC 파라미터 설정 (학습 속도를 위해 간소화)
    sampling_params = {
        'draws': 500,
        'tune': 500,
        'chains': 2,
        'target_accept': 0.95
    }
    
    # 모델 학습
    logger.info("모델 학습 중...")
    model.fit(train_data, sampling_params=sampling_params)
    
    # 테스트 데이터 예측
    logger.info("테스트 데이터에 대한 예측 수행 중...")
    forecast_test, lower_test, upper_test = model.predict(n_forecast=len(test_data))
    
    # 평가 지표 계산
    if hasattr(model, 'evaluate'):
        metrics = model.evaluate(test_data)
        logger.info(f"평가 지표: {metrics}")
    
    # 미래 예측
    logger.info(f"향후 {n_forecast}일 예측 중...")
    forecast, lower, upper = model.predict(n_forecast=n_forecast)
    
    # 결과 시각화 - 테스트 세트
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(train_data.index, train_data.values, label='학습 데이터', color='blue')
    plt.plot(test_data.index, test_data.values, label='실제 테스트 데이터', color='green')
    plt.plot(test_data.index, forecast_test, label='테스트 데이터 예측', color='red')
    plt.fill_between(test_data.index, lower_test, upper_test, color='red', alpha=0.2, label='95% 신뢰 구간')
    plt.title(f"{symbol} 가격 - 테스트 데이터 예측 ({model_type} 모델)")
    plt.legend()
    plt.grid(True)
    
    # 결과 시각화 - 미래 예측
    plt.subplot(2, 1, 2)
    
    # 전체 데이터 및 예측
    plt.plot(prices.index, prices.values, label='실제 데이터', color='blue')
    
    # 미래 날짜 생성
    future_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=n_forecast, freq='D')
    
    plt.plot(future_dates, forecast, label='미래 예측', color='red')
    plt.fill_between(future_dates, lower, upper, color='red', alpha=0.2, label='95% 신뢰 구간')
    plt.title(f"{symbol} 가격 - 향후 {n_forecast}일 예측 ({model_type} 모델)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_{model_type}_prediction.png")
    plt.show()
    
    # 모델별 추가 시각화
    if model_type == "structural" and hasattr(model, 'plot_components'):
        logger.info("구조적 모델 컴포넌트 시각화 중...")
        fig = model.plot_components()
        plt.savefig(f"{symbol}_{model_type}_components.png")
        plt.show()
    
    # 예측 결과 출력
    logger.info(f"\n향후 {n_forecast}일 {symbol} 가격 예측:")
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': forecast,
        'lower_95': lower,
        'upper_95': upper
    })
    forecast_df.set_index('date', inplace=True)
    print(forecast_df)

if __name__ == "__main__":
    # 자기회귀 모델로 암호화폐 가격 예측
    run_crypto_price_prediction(
        model_type="ar",
        symbol="BTCUSDT",
        n_forecast=30,
        model_params={
            "ar_order": 5,
            "seasonality": True,
            "num_seasons": 7
        }
    ) 