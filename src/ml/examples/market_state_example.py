#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
시장 상태 분류 예제 스크립트
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.classification.market_state_classifier import MarketStateClassifier
from src.data.collectors.binance_collector import BinanceDataCollector
from src.utils.logger import setup_logger

# 로거 설정
logger = setup_logger('market_state_example')

def prepare_price_data(symbol='BTCUSDT', interval='1h', start_date=None, end_date=None, limit=1000):
    """
    시장 데이터 준비
    """
    logger.info(f"데이터 수집 시작: {symbol}, 간격: {interval}")
    
    # 날짜 설정
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=90)  # 기본 90일 데이터 (충분한 학습 데이터 확보)
        
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

def run_market_state_analysis():
    """
    시장 상태 분류 실행
    """
    logger.info("시장 상태 분류 실행 시작")
    
    # 1. 시장 데이터 준비
    df = prepare_price_data(symbol='BTCUSDT', interval='4h')  # 4시간봉 데이터로 학습
    
    # 2. 분류 모델 설정
    config = {
        'class_thresholds': {
            'bullish': 0.03,    # 3% 이상 상승
            'bearish': -0.03,   # 3% 이상 하락
            'sideways': 0.01    # -1% ~ 1% 횡보
        },
        'prediction_period': 24,  # 24시간(6개 4시간봉) 후 예측
        'model_type': 'random_forest',  # 랜덤 포레스트 모델 사용
        'feature_window': 14,
        'model_save_path': 'models/market_state/'
    }
    
    # 3. 분류기 초기화 및 데이터 준비
    classifier = MarketStateClassifier(config)
    features, labels = classifier.prepare_data(df)
    
    # 4. 모델 훈련
    metrics = classifier.train(features, labels)
    
    # 5. 성능 출력
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
    
    # 6. 모델 저장
    model_path = classifier.save()
    
    # 7. 최근 데이터에 대한 예측
    recent_data = df.iloc[-30:]  # 최근 30개 데이터 포인트
    predictions = classifier.predict(recent_data[features.columns])
    
    # 8. 결과 시각화
    plt.figure(figsize=(15, 8))
    
    # 가격 차트
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(recent_data.index, recent_data['close'], label='BTC 가격')
    ax1.set_title('BTC 가격 및 시장 상태 예측')
    ax1.set_ylabel('가격 (USDT)')
    ax1.legend()
    ax1.grid(True)
    
    # 예측 결과
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    
    # 각 상태별 색상 매핑
    color_map = {
        'bullish': 'green',
        'bearish': 'red',
        'sideways': 'blue',
        'neutral': 'gray'
    }
    
    # 예측 상태를 색상으로 시각화
    for state in color_map.keys():
        mask = predictions['predicted_state'] == state
        if mask.any():
            ax2.scatter(
                predictions.index[mask],
                [0.5] * mask.sum(),  # Y 위치 (중앙에 배치)
                color=color_map[state],
                label=state,
                s=100,
                marker='o'
            )
    
    # 확률 그래프 (있는 경우)
    if 'prob_bullish' in predictions.columns:
        ax3 = ax2.twinx()
        ax3.plot(predictions.index, predictions['prob_bullish'], 'g--', alpha=0.5, label='상승 확률')
        ax3.plot(predictions.index, predictions['prob_bearish'], 'r--', alpha=0.5, label='하락 확률')
        ax3.set_ylabel('확률')
        ax3.legend(loc='upper right')
    
    ax2.set_xlabel('시간')
    ax2.set_ylabel('시장 상태')
    ax2.set_yticks([])
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 9. 미래 예측 (마지막 데이터 포인트의 상태)
    last_prediction = predictions.iloc[-1]
    logger.info(f"현재 시장 상태 예측: {last_prediction['predicted_state']}")
    
    if 'prob_bullish' in last_prediction:
        logger.info(f"  상승 확률: {last_prediction['prob_bullish']:.2f}")
        logger.info(f"  하락 확률: {last_prediction['prob_bearish']:.2f}")
        logger.info(f"  횡보 확률: {last_prediction['prob_sideways']:.2f}")
        logger.info(f"  중립 확률: {last_prediction['prob_neutral']:.2f}")
    
    logger.info("시장 상태 분류 실행 완료")
    
    return classifier

if __name__ == '__main__':
    run_market_state_analysis() 