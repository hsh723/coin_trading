#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
암호화폐 트레이딩 시스템 머신러닝 모듈 실행 스크립트
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta
import json
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/ml_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 모듈 임포트
from ml.training_pipeline import ModelTrainingPipeline
from ml.forecasting.model_factory import TimeSeriesPredictorFactory, ModelType
from ml.classification.market_state_classifier import MarketStateClassifier
from data.collectors.binance_collector import BinanceDataCollector

def prepare_market_data(symbol='BTCUSDT', interval='1h', days=60):
    """
    시장 데이터 준비 함수
    
    Args:
        symbol: 암호화폐 심볼
        interval: 시간 간격
        days: 데이터 일수
        
    Returns:
        준비된 데이터프레임
    """
    logger.info(f"시장 데이터 수집 시작 (심볼: {symbol}, 간격: {interval}, 기간: {days}일)")
    
    # 날짜 범위 설정
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
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
    기술적 지표 추가 함수
    
    Args:
        df: 원시 데이터프레임
        
    Returns:
        지표가 추가된 데이터프레임
    """
    logger.info("기술적 지표 추가 중...")
    
    # 이동평균선
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma14'] = df['close'].rolling(window=14).mean()
    df['ma30'] = df['close'].rolling(window=30).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    
    # 볼린저 밴드
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['std20'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['ma20'] + (df['std20'] * 2)
    df['lower_band'] = df['ma20'] - (df['std20'] * 2)
    df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['ma20']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 기타 지표
    df['atr'] = calculate_atr(df, 14)
    df['cci'] = calculate_cci(df, 20)
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, 14, 3)
    
    # 가격 모멘텀
    df['price_momentum'] = df['close'].pct_change(periods=10)
    
    # 거래량 지표
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # 결측치 제거
    df.dropna(inplace=True)
    
    logger.info(f"지표 추가 후 데이터: {len(df)} 행")
    
    return df

def calculate_atr(df, period=14):
    """ATR(Average True Range) 계산"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    return true_range.rolling(period).mean()

def calculate_cci(df, period=20):
    """CCI(Commodity Channel Index) 계산"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    ma = typical_price.rolling(period).mean()
    mean_deviation = (typical_price - ma).abs().rolling(period).mean()
    
    return (typical_price - ma) / (0.015 * mean_deviation)

def calculate_stochastic(df, k_period=14, d_period=3):
    """스토캐스틱 오실레이터 계산"""
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    
    stoch_k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d

def run_price_prediction(df, config=None):
    """
    가격 예측 모델 실행
    
    Args:
        df: 준비된 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        훈련 결과 객체
    """
    logger.info("가격 예측 모델 실행 시작")
    
    # 기본 설정
    default_config = {
        'sequence_length': 48,
        'target_column': 'close',
        'model_type': ModelType.LSTM,
        'batch_size': 32,
        'epochs': 100,
        'patience': 15,
        'optimization_trials': 20,
        'test_size': 0.15,
        'validation_size': 0.15,
        'forecast_horizon': 24,
        'visualize_results': True,
        'save_model': True,
        'model_save_path': 'models/price_prediction/'
    }
    
    # 설정 병합
    if config:
        config = {**default_config, **config}
    else:
        config = default_config
    
    # 특성 선택
    features = [
        'close', 'volume', 'ma7', 'ma14', 'ma30', 'rsi', 'macd', 'macd_hist',
        'bb_width', 'cci', 'stoch_k', 'price_momentum', 'volume_ratio'
    ]
    
    # 학습 파이프라인 생성 및 실행
    pipeline = ModelTrainingPipeline(config)
    result = pipeline.run_pipeline(df, features=features, target=config['target_column'])
    
    # 결과 출력
    logger.info("모델 훈련 결과:")
    for key, value in result.metrics.items():
        logger.info(f"  {key}: {value:.6f}")
    
    if result.best_params:
        logger.info("최적 하이퍼파라미터:")
        for key, value in result.best_params.items():
            logger.info(f"  {key}: {value}")
    
    # 미래 예측
    model = result.model
    last_sequence = df[features].iloc[-config['sequence_length']:].values
    future_pred = model.forecast(last_sequence, horizon=config['forecast_horizon'])
    
    logger.info(f"향후 {config['forecast_horizon']}시간 예측 결과:")
    for i, pred in enumerate(future_pred):
        logger.info(f"  {i+1}시간 후: {pred[0]:.2f} USD")
    
    logger.info("가격 예측 모델 실행 완료")
    
    return result

def run_market_state_classification(df, config=None):
    """
    시장 상태 분류 모델 실행
    
    Args:
        df: 준비된 데이터프레임
        config: 설정 딕셔너리
        
    Returns:
        훈련된 분류기 객체
    """
    logger.info("시장 상태 분류 모델 실행 시작")
    
    # 기본 설정
    default_config = {
        'class_thresholds': {
            'bullish': 0.03,    # 3% 이상 상승
            'bearish': -0.03,   # 3% 이상 하락
            'sideways': 0.01    # -1% ~ 1% 횡보
        },
        'prediction_period': 24,  # 24시간 후 예측
        'model_type': 'random_forest',
        'feature_window': 14,
        'model_save_path': 'models/market_state/'
    }
    
    # 설정 병합
    if config:
        config = {**default_config, **config}
    else:
        config = default_config
    
    # 분류기 초기화 및 데이터 준비
    classifier = MarketStateClassifier(config)
    features, labels = classifier.prepare_data(df)
    
    # 모델 훈련
    metrics = classifier.train(features, labels)
    
    # 성능 출력
    logger.info("모델 성능:")
    logger.info(f"  정확도: {metrics['accuracy']:.4f}")
    
    # 각 클래스별 성능
    for state in ['bullish', 'bearish', 'sideways', 'neutral']:
        if state in metrics['report']:
            precision = metrics['report'][state]['precision']
            recall = metrics['report'][state]['recall']
            f1 = metrics['report'][state]['f1-score']
            support = metrics['report'][state]['support']
            logger.info(f"  {state}: 정밀도={precision:.4f}, 재현율={recall:.4f}, F1={f1:.4f}, 샘플수={support}")
    
    # 모델 저장
    model_path = classifier.save()
    
    # 최근 데이터에 대한 예측
    recent_data = df.iloc[-30:]  # 최근 30개 데이터 포인트
    predictions = classifier.predict(recent_data[features.columns])
    
    # 미래 예측 (마지막 데이터 포인트의 상태)
    last_prediction = predictions.iloc[-1]
    logger.info(f"현재 시장 상태 예측: {last_prediction['predicted_state']}")
    
    if 'prob_bullish' in last_prediction:
        logger.info(f"  상승 확률: {last_prediction['prob_bullish']:.2f}")
        logger.info(f"  하락 확률: {last_prediction['prob_bearish']:.2f}")
        logger.info(f"  횡보 확률: {last_prediction['prob_sideways']:.2f}")
        logger.info(f"  중립 확률: {last_prediction['prob_neutral']:.2f}")
    
    logger.info("시장 상태 분류 모델 실행 완료")
    
    return classifier

def run_integrated_ml_system(args):
    """
    통합 머신러닝 시스템 실행
    
    Args:
        args: 명령행 인자
    """
    logger.info("암호화폐 트레이딩 시스템 머신러닝 모듈 실행")
    
    # 데이터 준비
    df = prepare_market_data(
        symbol=args.symbol,
        interval=args.interval,
        days=args.days
    )
    
    # 기술적 지표 추가
    df = add_technical_indicators(df)
    
    # 결과 저장 디렉토리 생성
    os.makedirs('results', exist_ok=True)
    
    # 모델 실행
    results = {}
    
    if args.price_prediction or args.all:
        # 가격 예측 설정
        price_config = {
            'sequence_length': args.sequence_length,
            'forecast_horizon': args.forecast_horizon,
            'model_type': getattr(ModelType, args.model_type),
            'epochs': args.epochs,
            'optimization_trials': args.trials
        }
        
        # 가격 예측 모델 실행
        price_result = run_price_prediction(df, price_config)
        results['price_prediction'] = {
            'metrics': price_result.metrics,
            'best_params': price_result.best_params
        }
    
    if args.market_state or args.all:
        # 시장 상태 분류 설정
        state_config = {
            'prediction_period': args.prediction_period,
            'model_type': args.classifier_type,
            'class_thresholds': {
                'bullish': args.bullish_threshold,
                'bearish': -args.bearish_threshold,
                'sideways': args.sideways_threshold
            }
        }
        
        # 시장 상태 분류 모델 실행
        state_classifier = run_market_state_classification(df, state_config)
        
        # 결과에 분류기 정보 추가
        feature_importances = getattr(state_classifier.model, 'feature_importances_', None)
        if feature_importances is not None:
            feature_names = state_classifier.model.feature_names_in_
            importances_dict = {name: float(imp) for name, imp in zip(feature_names, feature_importances)}
            results['market_state'] = {
                'feature_importances': importances_dict
            }
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"results/ml_system_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"결과가 저장되었습니다: {result_file}")
    logger.info("암호화폐 트레이딩 시스템 머신러닝 모듈 실행 완료")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='암호화폐 트레이딩 시스템 머신러닝 모듈')
    
    # 데이터 관련 인자
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='암호화폐 심볼')
    parser.add_argument('--interval', type=str, default='1h', help='시간 간격')
    parser.add_argument('--days', type=int, default=90, help='데이터 일수')
    
    # 실행 모드 인자
    parser.add_argument('--all', action='store_true', help='모든 모델 실행')
    parser.add_argument('--price-prediction', action='store_true', help='가격 예측 모델 실행')
    parser.add_argument('--market-state', action='store_true', help='시장 상태 분류 모델 실행')
    
    # 가격 예측 모델 인자
    parser.add_argument('--model-type', type=str, default='LSTM', 
                       choices=['LSTM', 'GRU', 'SIMPLE_RNN', 'CNN_LSTM', 'BIDIRECTIONAL_LSTM'],
                       help='시계열 모델 유형')
    parser.add_argument('--sequence-length', type=int, default=48, help='입력 시퀀스 길이')
    parser.add_argument('--forecast-horizon', type=int, default=24, help='예측 기간')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에포크 수')
    parser.add_argument('--trials', type=int, default=20, help='하이퍼파라미터 최적화 시도 횟수')
    
    # 시장 상태 분류 모델 인자
    parser.add_argument('--classifier-type', type=str, default='random_forest',
                       choices=['random_forest', 'xgboost', 'gradient_boosting', 'svm'],
                       help='분류 모델 유형')
    parser.add_argument('--prediction-period', type=int, default=24, help='분류 예측 기간')
    parser.add_argument('--bullish-threshold', type=float, default=0.03, help='상승장 임계값')
    parser.add_argument('--bearish-threshold', type=float, default=0.03, help='하락장 임계값')
    parser.add_argument('--sideways-threshold', type=float, default=0.01, help='횡보장 임계값')
    
    args = parser.parse_args()
    
    # 기본적으로 하나라도 선택하지 않으면 모두 실행
    if not (args.all or args.price_prediction or args.market_state):
        args.all = True
    
    # 시스템 실행
    run_integrated_ml_system(args) 