#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
암호화폐 가격 예측 예제 스크립트
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.training_pipeline import ModelTrainingPipeline
from src.ml.forecasting.model_factory import ModelType
from src.data.collectors.binance_collector import BinanceDataCollector
from src.utils.logger import setup_logger

# 로거 설정
logger = setup_logger('price_prediction_example')

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
    
    # 누락된 값 제거
    df.dropna(inplace=True)
    
    logger.info(f"지표 추가 후 데이터: {len(df)} 행")
    
    return df

def run_price_prediction():
    """
    가격 예측 모델 실행
    """
    logger.info("가격 예측 모델 실행 시작")
    
    # 1. 데이터 준비
    df = prepare_price_data(symbol='BTCUSDT', interval='1h')
    df = add_technical_indicators(df)
    
    # 2. 특성 선택
    features = ['close', 'volume', 'ma7', 'ma14', 'ma30', 'rsi', 'macd', 'macd_hist']
    target_col = 'close'
    
    # 3. 모델 설정
    config = {
        'sequence_length': 48,  # 48시간(2일) 데이터로 예측
        'target_column': target_col,
        'model_type': ModelType.LSTM,  # LSTM 모델 선택
        'batch_size': 32,
        'epochs': 100,
        'patience': 15,
        'optimization_trials': 20,
        'test_size': 0.15,
        'validation_size': 0.15,
        'forecast_horizon': 24,  # 24시간(1일) 예측
        'visualize_results': True,
        'save_model': True,
        'model_save_path': 'models/price_prediction/'
    }
    
    # 4. 학습 파이프라인 생성 및 실행
    pipeline = ModelTrainingPipeline(config)
    result = pipeline.run_pipeline(df, features=features, target=target_col)
    
    # 5. 결과 출력
    logger.info("모델 훈련 결과:")
    for key, value in result.metrics.items():
        logger.info(f"  {key}: {value:.6f}")
    
    if result.best_params:
        logger.info("최적 하이퍼파라미터:")
        for key, value in result.best_params.items():
            logger.info(f"  {key}: {value}")
    
    # 6. 미래 24시간 예측
    model = result.model
    last_sequence = df[features].iloc[-config['sequence_length']:].values
    future_pred = model.forecast(last_sequence, horizon=config['forecast_horizon'])
    
    logger.info(f"향후 {config['forecast_horizon']}시간 예측 결과:")
    for i, pred in enumerate(future_pred):
        logger.info(f"  {i+1}시간 후: {pred[0]:.2f} USD")
    
    logger.info("가격 예측 모델 실행 완료")
    
    return result

if __name__ == '__main__':
    run_price_prediction() 