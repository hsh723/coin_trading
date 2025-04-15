#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
시계열 모델 온라인 학습 예제 스크립트
실시간으로 수신되는 데이터를 활용하여 모델을 지속적으로 업데이트합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import logging
import json
import argparse

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.online.time_series_online_learner import OnlineLearner
from src.ml.forecasting.model_factory import ModelType
from src.data.collectors.binance_collector import BinanceDataCollector
from src.utils.logger import setup_logger

# 로거 설정
logger = setup_logger('online_learning_example')

def prepare_initial_data(symbol='BTCUSDT', interval='1h', days=60):
    """
    초기 학습 데이터 준비
    """
    logger.info(f"초기 학습 데이터 수집 시작: {symbol}, 간격: {interval}, 기간: {days}일")
    
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
    
    logger.info(f"수집된 초기 데이터: {len(df)} 행")
    
    return df

def add_technical_indicators(df):
    """
    기술적 지표 추가
    """
    df_copy = df.copy()
    
    # 이동평균선
    df_copy['ma7'] = df_copy['close'].rolling(window=7).mean()
    df_copy['ma14'] = df_copy['close'].rolling(window=14).mean()
    df_copy['ma30'] = df_copy['close'].rolling(window=30).mean()
    
    # 볼린저 밴드
    df_copy['ma20'] = df_copy['close'].rolling(window=20).mean()
    df_copy['std20'] = df_copy['close'].rolling(window=20).std()
    df_copy['upper_band'] = df_copy['ma20'] + (df_copy['std20'] * 2)
    df_copy['lower_band'] = df_copy['ma20'] - (df_copy['std20'] * 2)
    df_copy['bb_width'] = (df_copy['upper_band'] - df_copy['lower_band']) / df_copy['ma20']
    
    # RSI (상대강도지수)
    delta = df_copy['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_copy['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (이동평균수렴확산)
    exp1 = df_copy['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_copy['close'].ewm(span=26, adjust=False).mean()
    df_copy['macd'] = exp1 - exp2
    df_copy['signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
    df_copy['macd_hist'] = df_copy['macd'] - df_copy['signal']
    
    # 가격 모멘텀
    df_copy['momentum'] = df_copy['close'].pct_change(periods=10)
    
    # 거래량 변화율
    df_copy['volume_change'] = df_copy['volume'].pct_change()
    
    # ATR (평균 진폭)
    high_low = df_copy['high'] - df_copy['low']
    high_close = (df_copy['high'] - df_copy['close'].shift()).abs()
    low_close = (df_copy['low'] - df_copy['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df_copy['atr'] = true_range.rolling(14).mean()
    
    # 누락된 값 제거
    df_copy.dropna(inplace=True)
    
    return df_copy

def get_new_data(collector, symbol, interval, start_time):
    """
    새로운 데이터 수집
    """
    try:
        data = collector.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=int(start_time.timestamp() * 1000),
            end_str=int(datetime.now().timestamp() * 1000)
        )
        
        if not data:
            return None
            
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
        
        return df
        
    except Exception as e:
        logger.error(f"새 데이터 수집 중 오류 발생: {e}")
        return None

def simulate_online_learning(symbol='BTCUSDT', interval='1h', initial_days=30, 
                            model_path=None, simulation_days=7, update_hours=6):
    """
    온라인 학습 시뮬레이션 실행
    
    Args:
        symbol: 거래 심볼
        interval: 데이터 간격
        initial_days: 초기 학습 데이터 기간(일)
        model_path: 기존 모델 경로 (없으면 새로 생성)
        simulation_days: 시뮬레이션 기간(일)
        update_hours: 업데이트 주기(시간)
    """
    logger.info(f"온라인 학습 시뮬레이션 시작: {symbol}, 시뮬레이션 기간: {simulation_days}일, 업데이트 주기: {update_hours}시간")
    
    # 초기 데이터 수집
    end_date = datetime.now()
    initial_end_date = end_date - timedelta(days=simulation_days)
    
    # 데이터 수집
    collector = BinanceDataCollector()
    
    # 온라인 학습 시뮬레이션을 위한 전체 데이터 수집
    start_date = end_date - timedelta(days=initial_days + simulation_days)
    
    all_data = collector.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_date.strftime('%Y-%m-%d'),
        end_str=end_date.strftime('%Y-%m-%d')
    )
    
    # 데이터프레임 변환
    df_all = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # 데이터 타입 변환
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_all[col] = df_all[col].astype(float)
    
    # 인덱스 설정
    df_all.set_index('timestamp', inplace=True)
    
    # 기술적 지표 추가
    df_all = add_technical_indicators(df_all)
    
    # 초기 학습 데이터와 시뮬레이션 데이터 분리
    initial_data = df_all[df_all.index < initial_end_date].copy()
    simulation_data = df_all[df_all.index >= initial_end_date].copy()
    
    logger.info(f"초기 학습 데이터: {len(initial_data)} 행")
    logger.info(f"시뮬레이션 데이터: {len(simulation_data)} 행")
    
    # 특성 선택
    features = [
        'close', 'volume', 'ma7', 'ma14', 'ma30', 
        'rsi', 'macd', 'macd_hist', 'bb_width', 
        'momentum', 'volume_change', 'atr'
    ]
    
    # 온라인 학습기 설정
    online_config = {
        'model_type': 'ensemble',
        'model_path': model_path,
        'update_frequency': update_hours,  # 시간 단위
        'sequence_length': 48,
        'n_features': len(features),
        'target_column': 'close',
        'model_save_path': 'models/online/',
        'ensemble_config': {
            'sequence_length': 48,
            'n_features': len(features),
            'ensemble_method': 'optimal',
            'model_types': [ModelType.LSTM, ModelType.GRU, ModelType.BIDIRECTIONAL_LSTM]
        }
    }
    
    # 온라인 학습기 초기화
    learner = OnlineLearner(online_config)
    
    # 초기 학습 (이미 모델이 있으면 건너뜀)
    if model_path is None or not os.path.exists(model_path):
        logger.info("초기 학습 시작...")
        update_result = learner.update(initial_data, features)
        
        if update_result:
            logger.info(f"초기 학습 완료 - RMSE: {update_result.metrics_after['rmse']:.6f}")
        else:
            logger.error("초기 학습 실패")
            return
    
    # 시뮬레이션 결과 저장용
    simulation_results = []
    prediction_history = []
    
    # 시뮬레이션 시작
    logger.info("시뮬레이션 시작...")
    
    # 시뮬레이션을 위한 시간 간격 계산
    time_intervals = []
    current_time = initial_end_date
    while current_time < end_date:
        time_intervals.append(current_time)
        current_time += timedelta(hours=update_hours)
    
    # 각 시점마다 가용한 데이터로 업데이트 및 예측 수행
    for i, current_time in enumerate(time_intervals):
        logger.info(f"시뮬레이션 진행: {i+1}/{len(time_intervals)} - {current_time}")
        
        # 현재 시점까지의 데이터
        current_data = df_all[df_all.index <= current_time].copy()
        
        # 온라인 업데이트
        update_result = learner.update(current_data, features)
        
        if update_result:
            logger.info(f"모델 업데이트 완료 - RMSE: {update_result.metrics_after['rmse']:.6f} (변화: {update_result.performance_change['rmse_change']*100:.2f}%)")
            
            # 결과 저장
            simulation_results.append({
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'metrics_before': update_result.metrics_before,
                'metrics_after': update_result.metrics_after,
                'performance_change': update_result.performance_change,
                'data_points': len(current_data)
            })
        
        # 미래 24시간 예측
        try:
            future_horizon = 24
            future_pred = learner.forecast(current_data[features], horizon=future_horizon)
            
            # 예측 결과 저장
            next_timestamps = []
            actual_values = []
            
            for h in range(future_horizon):
                pred_time = current_time + timedelta(hours=h+1)
                pred_value = float(future_pred[h][0])
                
                next_timestamps.append(pred_time)
                
                # 실제 값 확인 (가능한 경우)
                if pred_time in df_all.index:
                    actual_values.append(float(df_all.loc[pred_time, 'close']))
                else:
                    actual_values.append(None)
                
                prediction_history.append({
                    'prediction_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'target_time': pred_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'horizon': h+1,
                    'predicted_value': pred_value,
                    'actual_value': actual_values[-1]
                })
            
            logger.info(f"예측 완료: 다음 24시간 첫번째 값 = {future_pred[0][0]:.2f}, 마지막 값 = {future_pred[-1][0]:.2f}")
        
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}")
    
    # 시뮬레이션 결과 시각화
    plt.figure(figsize=(15, 10))
    
    # 1. 성능 변화 그래프
    plt.subplot(2, 1, 1)
    
    times = [datetime.strptime(r['timestamp'], '%Y-%m-%d %H:%M:%S') for r in simulation_results]
    rmse_before = [r['metrics_before']['rmse'] for r in simulation_results]
    rmse_after = [r['metrics_after']['rmse'] for r in simulation_results]
    
    plt.plot(times, rmse_before, 'r--', label='업데이트 전 RMSE')
    plt.plot(times, rmse_after, 'g-', label='업데이트 후 RMSE')
    
    plt.title('온라인 학습 성능 변화')
    plt.xlabel('시간')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    # 2. 예측 정확도 시각화
    plt.subplot(2, 1, 2)
    
    # 1시간, 4시간, 12시간, 24시간 예측 정확도 계산
    horizons_to_plot = [1, 4, 12, 24]
    accuracy_data = {}
    
    for horizon in horizons_to_plot:
        accuracy_data[horizon] = []
        
        for i, current_time in enumerate(time_intervals):
            # 현재 시점에서 horizon 시간 후의 예측
            horizon_preds = [p for p in prediction_history 
                            if datetime.strptime(p['prediction_time'], '%Y-%m-%d %H:%M:%S') == current_time 
                            and p['horizon'] == horizon
                            and p['actual_value'] is not None]
            
            if horizon_preds:
                pred = horizon_preds[0]
                pred_error = abs(pred['predicted_value'] - pred['actual_value']) / pred['actual_value'] * 100
                accuracy_data[horizon].append((current_time, 100 - pred_error))
    
    # 예측 정확도 그래프
    for horizon in horizons_to_plot:
        if accuracy_data[horizon]:
            times, accuracies = zip(*accuracy_data[horizon])
            plt.plot(times, accuracies, label=f'{horizon}시간 예측 정확도')
    
    plt.title('예측 정확도 (100% - 상대오차%)')
    plt.xlabel('시간')
    plt.ylabel('정확도 (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 결과 저장
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 그래프 저장
    plt.savefig(f'results/online_learning_simulation_{timestamp}.png')
    
    # 데이터 저장
    with open(f'results/online_learning_results_{timestamp}.json', 'w') as f:
        json.dump({
            'simulation_config': {
                'symbol': symbol,
                'interval': interval,
                'initial_days': initial_days,
                'simulation_days': simulation_days,
                'update_hours': update_hours
            },
            'model_config': online_config,
            'simulation_results': simulation_results,
            'prediction_history': prediction_history
        }, f, indent=4)
    
    logger.info(f"시뮬레이션 완료. 결과 저장됨: results/online_learning_results_{timestamp}.json")
    
    return learner

def run_real_online_learning(symbol='BTCUSDT', interval='1h', model_path=None, 
                           update_hours=6, run_duration_hours=None):
    """
    실제 온라인 학습 실행
    
    Args:
        symbol: 거래 심볼
        interval: 데이터 간격
        model_path: 기존 모델 경로 (없으면 새로 생성)
        update_hours: 업데이트 주기(시간)
        run_duration_hours: 실행 기간(시간), None이면 무기한 실행
    """
    logger.info(f"실제 온라인 학습 시작: {symbol}, 업데이트 주기: {update_hours}시간")
    
    # 데이터 수집기 초기화
    collector = BinanceDataCollector()
    
    # 초기 데이터 준비
    initial_data = prepare_initial_data(symbol, interval, days=60)
    initial_data = add_technical_indicators(initial_data)
    
    # 특성 선택
    features = [
        'close', 'volume', 'ma7', 'ma14', 'ma30', 
        'rsi', 'macd', 'macd_hist', 'bb_width', 
        'momentum', 'volume_change', 'atr'
    ]
    
    # 온라인 학습기 설정
    online_config = {
        'model_type': 'ensemble',
        'model_path': model_path,
        'update_frequency': update_hours,  # 시간 단위
        'sequence_length': 48,
        'n_features': len(features),
        'target_column': 'close',
        'model_save_path': 'models/online/',
        'ensemble_config': {
            'sequence_length': 48,
            'n_features': len(features),
            'ensemble_method': 'optimal',
            'model_types': [ModelType.LSTM, ModelType.GRU, ModelType.BIDIRECTIONAL_LSTM]
        }
    }
    
    # 온라인 학습기 초기화
    learner = OnlineLearner(online_config)
    
    # 초기 학습 (이미 모델이 있으면 건너뜀)
    if model_path is None or not os.path.exists(model_path):
        logger.info("초기 학습 시작...")
        update_result = learner.update(initial_data, features)
        
        if update_result:
            logger.info(f"초기 학습 완료 - RMSE: {update_result.metrics_after['rmse']:.6f}")
        else:
            logger.error("초기 학습 실패")
            return
    
    # 예측 결과 저장용
    prediction_history = []
    
    # 마지막 데이터 시간
    last_data_time = initial_data.index[-1]
    
    # 종료 시간 계산
    start_time = datetime.now()
    end_time = None
    if run_duration_hours:
        end_time = start_time + timedelta(hours=run_duration_hours)
        logger.info(f"예상 종료 시간: {end_time}")
    
    # 실시간 실행
    try:
        while True:
            current_time = datetime.now()
            
            # 실행 기간 확인
            if end_time and current_time > end_time:
                logger.info(f"설정된 실행 기간({run_duration_hours}시간)이 경과하여 종료합니다.")
                break
            
            # 새 데이터 수집
            logger.info(f"새 데이터 수집 중... (마지막 데이터: {last_data_time})")
            new_data = get_new_data(collector, symbol, interval, last_data_time)
            
            if new_data is not None and not new_data.empty:
                logger.info(f"새 데이터 수집 완료: {len(new_data)} 행")
                
                # 기술적 지표 추가
                new_data = add_technical_indicators(new_data)
                
                # 전체 데이터 업데이트
                full_data = pd.concat([initial_data, new_data])
                full_data = full_data[~full_data.index.duplicated(keep='last')]
                
                # 중복 없는 새 데이터만 유지
                new_rows = len(full_data) - len(initial_data)
                if new_rows > 0:
                    logger.info(f"새로운 데이터 행: {new_rows}")
                    
                    # 온라인 업데이트
                    update_result = learner.update(full_data, features)
                    
                    if update_result:
                        logger.info(f"모델 업데이트 완료 - RMSE: {update_result.metrics_after['rmse']:.6f} (변화: {update_result.performance_change['rmse_change']*100:.2f}%)")
                    
                    # 미래 예측
                    try:
                        future_horizon = 24
                        future_pred = learner.forecast(full_data[features], horizon=future_horizon)
                        
                        # 결과 출력
                        logger.info(f"향후 {future_horizon}시간 예측 결과:")
                        for i, pred in enumerate(future_pred):
                            logger.info(f"  {i+1}시간 후: {pred[0]:.2f} USD")
                        
                        # 예측 저장
                        for h in range(future_horizon):
                            pred_time = full_data.index[-1] + timedelta(hours=h+1)
                            prediction_history.append({
                                'prediction_time': full_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                                'target_time': pred_time.strftime('%Y-%m-%d %H:%M:%S'),
                                'horizon': h+1,
                                'predicted_value': float(future_pred[h][0])
                            })
                    except Exception as e:
                        logger.error(f"예측 중 오류 발생: {e}")
                    
                    # 데이터 업데이트
                    initial_data = full_data
                    last_data_time = initial_data.index[-1]
                else:
                    logger.info("새로운 데이터가 없습니다.")
            
            # 다음 업데이트까지 대기
            # 최소 1분 ~ 최대 update_hours 시간
            next_update_seconds = min(update_hours * 3600, max(60, update_hours * 3600 / 10))
            logger.info(f"다음 업데이트까지 {next_update_seconds/60:.1f}분 대기...")
            time.sleep(next_update_seconds)
    
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
    finally:
        # 결과 저장
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 예측 기록 저장
        with open(f'results/online_predictions_{timestamp}.json', 'w') as f:
            json.dump({
                'config': {
                    'symbol': symbol,
                    'interval': interval,
                    'update_hours': update_hours
                },
                'predictions': prediction_history,
                'run_duration': {
                    'start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'hours': (datetime.now() - start_time).total_seconds() / 3600
                }
            }, f, indent=4)
        
        logger.info(f"예측 결과 저장됨: results/online_predictions_{timestamp}.json")
    
    return learner

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='시계열 모델 온라인 학습 예제')
    parser.add_argument('--mode', type=str, choices=['simulation', 'real'], default='simulation',
                       help='실행 모드: simulation(시뮬레이션) 또는 real(실제 온라인 학습)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='거래 심볼 (기본값: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h',
                       help='데이터 간격 (기본값: 1h)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='기존 모델 경로 (없으면 새로 생성)')
    parser.add_argument('--initial_days', type=int, default=30,
                       help='초기 학습 데이터 기간(일) (기본값: 30)')
    parser.add_argument('--simulation_days', type=int, default=7,
                       help='시뮬레이션 기간(일) (기본값: 7)')
    parser.add_argument('--update_hours', type=int, default=6,
                       help='업데이트 주기(시간) (기본값: 6)')
    parser.add_argument('--run_hours', type=int, default=None,
                       help='실행 기간(시간), 지정하지 않으면 무기한 실행')
    
    args = parser.parse_args()
    
    if args.mode == 'simulation':
        # 시뮬레이션 모드
        simulate_online_learning(
            symbol=args.symbol,
            interval=args.interval,
            initial_days=args.initial_days,
            model_path=args.model_path,
            simulation_days=args.simulation_days,
            update_hours=args.update_hours
        )
    else:
        # 실제 온라인 학습 모드
        run_real_online_learning(
            symbol=args.symbol,
            interval=args.interval,
            model_path=args.model_path,
            update_hours=args.update_hours,
            run_duration_hours=args.run_hours
        )

if __name__ == '__main__':
    main() 