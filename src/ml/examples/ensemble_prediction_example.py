#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
앙상블 모델을 사용한 암호화폐 가격 예측 예제 스크립트
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.ensemble.time_series_ensemble import TimeSeriesEnsemble
from src.ml.forecasting.model_factory import ModelType
from src.data.collectors.binance_collector import BinanceDataCollector
from src.utils.logger import setup_logger

# 로거 설정
logger = setup_logger('ensemble_prediction_example')

def prepare_price_data(symbol='BTCUSDT', interval='1h', start_date=None, end_date=None, limit=1000):
    """
    시장 데이터 준비
    """
    logger.info(f"데이터 수집 시작: {symbol}, 간격: {interval}")
    
    # 날짜 설정
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=60)  # 기본 60일 데이터
        
    # 데이터 수집
    collector = BinanceDataCollector()
    data = collector.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_date.strftime('%Y-%m-%d'),
        end_str=end_date.strftime('%Y-%m-%d')
    )
    
    # 데이터프레임 변환
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # 데이터 타입 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # 인덱스 설정
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"수집된 데이터: {len(df)} 행")
    
    return df

def add_technical_indicators(df):
    """
    기술적 지표 추가
    """
    logger.info("기술적 지표 추가")
    
    # 이동평균선
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma14'] = df['close'].rolling(window=14).mean()
    df['ma30'] = df['close'].rolling(window=30).mean()
    
    # 볼린저 밴드
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['std20'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['ma20'] + (df['std20'] * 2)
    df['lower_band'] = df['ma20'] - (df['std20'] * 2)
    df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['ma20']
    
    # RSI (상대강도지수)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (이동평균수렴확산)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    
    # 스토캐스틱 오실레이터
    high_14 = df['high'].rolling(window=14).max()
    low_14 = df['low'].rolling(window=14).min()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # 가격 모멘텀
    df['momentum'] = df['close'].pct_change(periods=10)
    
    # 누락된 값 제거
    df.dropna(inplace=True)
    
    logger.info(f"지표 추가 후 데이터: {len(df)} 행")
    
    return df

def prepare_data_for_lstm(df, features, target_col='close', test_size=0.2, val_size=0.2, sequence_length=48):
    """
    LSTM 모델 훈련을 위한 데이터 준비
    """
    # 특성 및 타겟 데이터 추출
    data = df[features].copy()
    
    # 테스트 및 검증 세트 분할
    n = len(data)
    test_end = n
    test_start = n - int(n * test_size)
    val_end = test_start
    val_start = val_end - int(n * val_size)
    
    train_data = data.iloc[:val_start].copy()
    val_data = data.iloc[val_start:val_end].copy()
    test_data = data.iloc[test_start:].copy()
    
    logger.info(f"훈련 데이터: {len(train_data)} 샘플")
    logger.info(f"검증 데이터: {len(val_data)} 샘플")
    logger.info(f"테스트 데이터: {len(test_data)} 샘플")
    
    # 앙상블 모델 생성
    ensemble = TimeSeriesEnsemble({
        'sequence_length': sequence_length,
        'n_features': len(features),
        'ensemble_method': 'optimal',  # 최적 가중치 사용
        'optimization_metric': 'rmse',
        'model_types': [ModelType.LSTM, ModelType.GRU, ModelType.BIDIRECTIONAL_LSTM],
        'model_configs': [
            {
                'sequence_length': sequence_length,
                'n_features': len(features),
                'units1': 64,
                'units2': 32,
                'dropout_rate': 0.2
            },
            {
                'sequence_length': sequence_length,
                'n_features': len(features),
                'units1': 50,
                'units2': 25,
                'dropout_rate': 0.3
            },
            {
                'sequence_length': sequence_length,
                'n_features': len(features),
                'units': 64,
                'dropout_rate': 0.2
            }
        ]
    })
    
    # 모델 생성
    ensemble.build_models()
    
    # 데이터 준비
    # 첫 번째 모델을 사용하여 다변량 데이터 준비
    X_train, y_train = ensemble.models[0].prepare_multivariate_data(
        train_data, target_col=target_col
    )
    X_val, y_val = ensemble.models[0].prepare_multivariate_data(
        val_data, target_col=target_col
    )
    X_test, y_test = ensemble.models[0].prepare_multivariate_data(
        test_data, target_col=target_col
    )
    
    return ensemble, (X_train, y_train), (X_val, y_val), (X_test, y_test), test_data

def run_ensemble_prediction():
    """
    앙상블 예측 모델 실행
    """
    logger.info("앙상블 가격 예측 모델 실행 시작")
    
    # 1. 데이터 준비
    df = prepare_price_data(symbol='BTCUSDT', interval='1h')
    df = add_technical_indicators(df)
    
    # 2. 특성 선택
    features = [
        'close', 'volume', 'ma7', 'ma14', 'ma30', 'rsi', 
        'macd', 'macd_hist', 'bb_width', 'stoch_k', 'momentum'
    ]
    target_col = 'close'
    
    # 3. 훈련/검증/테스트 데이터 준비
    sequence_length = 48  # 2일 데이터로 예측
    ensemble, (X_train, y_train), (X_val, y_val), (X_test, y_test), test_data = prepare_data_for_lstm(
        df, features, target_col=target_col, 
        test_size=0.15, val_size=0.15, 
        sequence_length=sequence_length
    )
    
    # 4. 앙상블 모델 훈련
    logger.info("앙상블 모델 훈련 시작...")
    ensemble.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        patience=15
    )
    
    # 5. 모델 평가
    metrics = ensemble.evaluate(X_test, y_test)
    
    logger.info("앙상블 모델 성능:")
    logger.info(f"  RMSE: {metrics['rmse']:.6f}")
    logger.info(f"  MAE: {metrics['mae']:.6f}")
    logger.info(f"  R²: {metrics['r2']:.6f}")
    
    logger.info("개별 모델 성능:")
    for model_metric in metrics['individual_metrics']:
        logger.info(f"  모델 {model_metric['model_index']+1} ({model_metric['model_type']}): "
                   f"가중치={model_metric['weight']:.4f}, "
                   f"RMSE={model_metric['rmse']:.6f}, "
                   f"MAE={model_metric['mae']:.6f}")
    
    # 6. 결과 시각화
    ensemble.plot_predictions(X_test, y_test, title='비트코인 가격 앙상블 예측')
    
    # 7. 미래 예측
    forecast_horizon = 24  # 24시간 예측
    last_sequence = test_data[features].iloc[-sequence_length:].values
    
    future_pred = ensemble.forecast(last_sequence, horizon=forecast_horizon)
    
    # 결과 출력
    logger.info(f"향후 {forecast_horizon}시간 예측 결과:")
    for i, pred in enumerate(future_pred):
        logger.info(f"  {i+1}시간 후: {pred[0]:.2f} USD")
        
    # 미래 예측 시각화
    plt.figure(figsize=(15, 6))
    
    # 과거 실제 가격
    actual_prices = ensemble.models[0].scaler.inverse_transform(y_test)
    plt.plot(range(len(actual_prices)), actual_prices, label='과거 데이터', color='blue')
    
    # 미래 예측
    plt.plot(
        range(len(actual_prices), len(actual_prices) + forecast_horizon),
        future_pred,
        label='앙상블 예측', 
        color='red',
        linestyle='--'
    )
    
    # 분리선
    plt.axvline(x=len(actual_prices), color='green', linestyle='-.')
    
    plt.title('비트코인 가격 미래 예측')
    plt.xlabel('시간 (시간 단위)')
    plt.ylabel('가격 (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 8. 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    
    # 앙상블 가중치 저장
    weights = ensemble.get_ensemble_weights()
    weights_dict = {weight.model_name: weight.weight for weight in weights}
    
    results = {
        'model_type': 'ensemble',
        'timestamp': timestamp,
        'metrics': {
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2']
        },
        'individual_metrics': metrics['individual_metrics'],
        'ensemble_weights': weights_dict,
        'forecast': {
            'horizon': forecast_horizon,
            'values': [float(pred[0]) for pred in future_pred]
        }
    }
    
    # 결과 파일 저장
    result_path = f'results/ensemble_prediction_{timestamp}.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"결과가 저장되었습니다: {result_path}")
    logger.info("앙상블 가격 예측 모델 실행 완료")
    
    return ensemble, metrics

if __name__ == '__main__':
    run_ensemble_prediction() 