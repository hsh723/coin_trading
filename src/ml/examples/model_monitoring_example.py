#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML 성능 모니터링 시스템 예제 스크립트
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.ml.monitoring.performance_monitor import MLPerformanceMonitor
from src.utils.logger import setup_logger

# 로거 설정
logger = setup_logger('model_monitoring_example')

def generate_synthetic_data(n_samples=1000, n_features=10, noise_level=0.1, random_state=None):
    """
    합성 데이터 생성
    """
    np.random.seed(random_state)
    
    # 특성 생성
    X = np.random.randn(n_samples, n_features)
    
    # 특성간 상관관계 추가
    corr_matrix = np.eye(n_features)
    # 일부 특성간 상관관계 설정
    corr_matrix[0, 1] = 0.7
    corr_matrix[1, 0] = 0.7
    corr_matrix[2, 3] = 0.5
    corr_matrix[3, 2] = 0.5
    
    # Cholesky 분해를 이용한 상관 데이터 생성
    L = np.linalg.cholesky(corr_matrix)
    X = np.dot(X, L)
    
    # 타겟 변수 생성 (비선형 관계 포함)
    y = 3*X[:, 0]**2 + 2*X[:, 1] - 1.5*X[:, 2] + 0.5*X[:, 3]*X[:, 4] - X[:, 5]
    
    # 노이즈 추가
    y += noise_level * np.random.randn(n_samples)
    
    # 데이터프레임 변환
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    return df

def add_drift_to_data(original_data, feature_drifts=None, target_drift=0.0, random_state=None):
    """
    데이터에 드리프트 추가
    """
    np.random.seed(random_state)
    
    # 원본 데이터 복사
    drifted_data = original_data.copy()
    
    # 기본 특성 드리프트 설정
    if feature_drifts is None:
        feature_drifts = {
            'feature_0': 0.5,   # feature_0에 0.5 표준편차만큼 이동
            'feature_2': -0.3,  # feature_2에 -0.3 표준편차만큼 이동
            'feature_5': 0.7    # feature_5에 0.7 표준편차만큼 이동
        }
    
    # 특성 드리프트 적용
    for feature, drift in feature_drifts.items():
        if feature in drifted_data.columns:
            std = drifted_data[feature].std()
            drifted_data[feature] += drift * std
    
    # 타겟 드리프트 적용
    if target_drift != 0 and 'target' in drifted_data.columns:
        std = drifted_data['target'].std()
        drifted_data['target'] += target_drift * std
    
    return drifted_data

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='random_forest', random_state=None):
    """
    모델 훈련 및 평가
    """
    start_time = time.time()
    
    # 모델 선택
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    elif model_type == 'linear':
        model = LinearRegression()
    else:
        raise ValueError(f"지원하지 않는 모델 유형: {model_type}")
    
    # 모델 훈련
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 예측 및 평가
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # 평가 지표 계산
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return model, metrics, training_time, prediction_time

def run_monitoring_example():
    """
    ML 성능 모니터링 예제 실행
    """
    logger.info("ML 성능 모니터링 예제 시작")
    
    # 모니터링 시스템 초기화
    monitor = MLPerformanceMonitor(save_dir='logs/ml_performance',
                                 config={
                                     'alert_threshold': 0.1,
                                     'window_size': 3,
                                     'log_to_file': True,
                                     'drift_threshold': 0.01
                                 })
    
    # 1. 초기 데이터 생성
    logger.info("초기 데이터 생성")
    initial_data = generate_synthetic_data(n_samples=1000, n_features=10, 
                                         random_state=42)
    
    # 데이터 분할
    X = initial_data.drop('target', axis=1)
    y = initial_data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    # 2. 초기 모델 훈련 및 평가
    logger.info("초기 모델 훈련 및 평가")
    model_types = ['random_forest', 'gradient_boosting', 'linear']
    models = {}
    
    for model_type in model_types:
        model, metrics, training_time, prediction_time = train_and_evaluate_model(
            X_train_scaled, y_train, X_test_scaled, y_test,
            model_type=model_type, random_state=42
        )
        
        models[model_type] = model
        
        # 모니터링 시스템에 성능 기록
        monitor.log_metrics(
            model_id=model_type,
            metrics=metrics,
            data_size=len(X_test),
            timestamp=datetime.now() - timedelta(days=30),  # 30일 전 데이터로 가정
            training_time=training_time,
            prediction_time=prediction_time
        )
        
        logger.info(f"{model_type} 모델 초기 성능:")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  MAE: {metrics['mae']:.6f}")
        logger.info(f"  R²: {metrics['r2']:.6f}")
    
    # 3. 시간에 따른 성능 모니터링 시뮬레이션
    logger.info("시간에 따른 성능 모니터링 시뮬레이션")
    
    # 5개의 시점 시뮬레이션
    for i in range(1, 6):
        timestamp = datetime.now() - timedelta(days=30-i*5)  # 5일 간격으로 시뮬레이션
        logger.info(f"\n시점 {i}: {timestamp.strftime('%Y-%m-%d')}")
        
        # 데이터 드리프트 추가 (시간이 지남에 따라 점진적으로 증가)
        feature_drifts = {
            'feature_0': 0.1 * i,    # 점진적 증가
            'feature_2': -0.05 * i,  # 점진적 감소
            'feature_5': 0.15 * i    # 점진적 증가
        }
        
        # 드리프트된 데이터 생성
        drifted_data = add_drift_to_data(
            initial_data, 
            feature_drifts=feature_drifts,
            target_drift=0.05 * i,  # 타겟 변수도 점진적으로 드리프트
            random_state=42+i
        )
        
        # 데이터 분할
        X_d = drifted_data.drop('target', axis=1)
        y_d = drifted_data['target']
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
            X_d, y_d, test_size=0.2, random_state=42+i
        )
        
        # 스케일링
        X_train_d_scaled = pd.DataFrame(
            scaler.transform(X_train_d),
            columns=X_train_d.columns
        )
        X_test_d_scaled = pd.DataFrame(
            scaler.transform(X_test_d),
            columns=X_test_d.columns
        )
        
        # 드리프트 감지
        if i >= 3:  # 3번째 시점부터 드리프트 감지
            for model_type in model_types:
                logger.info(f"{model_type} 모델의 데이터 드리프트 감지")
                
                # 드리프트 감지
                drift_results = monitor.detect_data_drift(
                    model_id=model_type,
                    reference_data=X_train,  # 초기 훈련 데이터
                    current_data=X_train_d,  # 현재 드리프트된 데이터
                    features=X_train.columns.tolist()
                )
                
                # 결과 확인
                if drift_results['drift_detected']:
                    logger.info(f"  드리프트 감지됨! 영향받은 특성: {drift_results['drifted_features']}")
                    
                    # 드리프트된 특성 시각화
                    for feature in drift_results['drifted_features'][:2]:  # 최대 2개만 시각화
                        monitor.plot_feature_drift(
                            model_id=model_type,
                            reference_data=X_train,
                            current_data=X_train_d,
                            feature=feature,
                            show_plot=False
                        )
                else:
                    logger.info("  드리프트가 감지되지 않았습니다.")
        
        # 각 모델 재평가
        for model_type in model_types:
            # 기존 모델로 새 데이터 평가
            model = models[model_type]
            y_pred = model.predict(X_test_d_scaled)
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test_d, y_pred)),
                'mae': mean_absolute_error(y_test_d, y_pred),
                'r2': r2_score(y_test_d, y_pred)
            }
            
            # 모니터링 시스템에 성능 기록
            monitor.log_metrics(
                model_id=model_type,
                metrics=metrics,
                data_size=len(X_test_d),
                timestamp=timestamp,
                prediction_time=0.01
            )
            
            logger.info(f"{model_type} 모델 {i}번째 시점 성능:")
            logger.info(f"  RMSE: {metrics['rmse']:.6f}")
            logger.info(f"  MAE: {metrics['mae']:.6f}")
            logger.info(f"  R²: {metrics['r2']:.6f}")
            
            # 모델 재훈련 (3번째 시점부터)
            if i >= 3:
                _, new_metrics, training_time, prediction_time = train_and_evaluate_model(
                    X_train_d_scaled, y_train_d, X_test_d_scaled, y_test_d,
                    model_type=model_type, random_state=42+i
                )
                
                logger.info(f"{model_type} 모델 재훈련 후 성능:")
                logger.info(f"  RMSE: {new_metrics['rmse']:.6f}")
                logger.info(f"  MAE: {new_metrics['mae']:.6f}")
                logger.info(f"  R²: {new_metrics['r2']:.6f}")
    
    # 4. 성능 시각화
    logger.info("\n성능 시각화")
    
    # 각 모델의 RMSE 추세 시각화
    for model_type in model_types:
        monitor.plot_performance_trend(
            model_id=model_type,
            metric_name='rmse',
            show_plot=False
        )
        
        logger.info(f"{model_type} 모델의 RMSE 추세 그래프 생성 완료")
    
    # 모델 간 RMSE 비교
    monitor.plot_metrics_comparison(
        models=model_types,
        metric_name='rmse',
        show_plot=False
    )
    
    logger.info("모델 간 RMSE 비교 그래프 생성 완료")
    
    # 5. 성능 보고서 생성
    logger.info("\n성능 보고서 생성")
    
    for model_type in model_types:
        report = monitor.generate_performance_report(model_type)
        
        logger.info(f"{model_type} 모델 성능 보고서:")
        logger.info(f"  기간: {report['period']['start'].strftime('%Y-%m-%d')} ~ {report['period']['end'].strftime('%Y-%m-%d')}")
        logger.info(f"  데이터 포인트: {report['snapshots_count']}")
        
        # 주요 지표 요약
        for metric, stats in report['metrics_stats'].items():
            logger.info(f"  {metric}:")
            logger.info(f"    현재값: {stats['current']:.6f}")
            logger.info(f"    변화율: {stats['change']:.2%}")
            logger.info(f"    최소/최대: {stats['min']:.6f} / {stats['max']:.6f}")
        
        # 성능 저하 알림
        if report['degradation_alerts']:
            logger.info("  성능 저하 알림:")
            for metric, alert in report['degradation_alerts'].items():
                logger.info(f"    {metric}: {alert['previous']:.6f} -> {alert['current']:.6f} ({alert['change']:.2%} 변화)")
    
    logger.info("ML 성능 모니터링 예제 완료")

if __name__ == "__main__":
    run_monitoring_example() 