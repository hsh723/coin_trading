#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
적응형 특성 선택 메커니즘 예제 스크립트
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.ml.feature.adaptive_feature_selector import AdaptiveFeatureSelector
from src.utils.logger import setup_logger

# 로거 설정
logger = setup_logger('adaptive_feature_selection_example')

def generate_synthetic_data(n_samples=1000, n_features=50, 
                          n_informative=10, random_state=None):
    """
    합성 회귀 데이터 생성
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=random_state
    )
    
    # 특성 이름 생성
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # 데이터프레임 변환
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def add_noise_to_features(data, noise_level=0.1, random_state=None):
    """
    특성에 노이즈 추가
    """
    np.random.seed(random_state)
    
    df = data.copy()
    features = [col for col in df.columns if col != 'target']
    
    for feature in features:
        noise = np.random.normal(0, noise_level, size=len(df))
        df[feature] += noise
    
    return df

def simulate_feature_importance_change(data, important_features, 
                                     increase_factor=2.0, random_state=None):
    """
    시간 경과에 따른 특성 중요도 변화 시뮬레이션
    """
    np.random.seed(random_state)
    
    df = data.copy()
    target = df['target'].copy()
    
    # 지정된 중요 특성의 영향력 증가
    for feature in important_features:
        if feature in df.columns:
            # 특성과 타겟 간의 관계 강화
            contribution = df[feature] * increase_factor
            target += contribution
    
    # 타겟 업데이트
    df['target'] = target
    
    return df

def evaluate_model(X_train, y_train, X_test, y_test, selected_features=None):
    """
    선택된 특성으로 모델 학습 및 평가
    """
    if selected_features:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
    
    # 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = model.predict(X_test)
    
    # 평가 메트릭스
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_features': X_train.shape[1]
    }

def run_adaptive_feature_selection_example():
    """
    적응형 특성 선택 메커니즘 예제 실행
    """
    logger.info("적응형 특성 선택 메커니즘 예제 시작")
    
    # 1. 합성 데이터 생성
    logger.info("합성 데이터 생성")
    n_total_features = 50
    n_informative = 10
    
    initial_data = generate_synthetic_data(
        n_samples=2000,
        n_features=n_total_features,
        n_informative=n_informative,
        random_state=42
    )
    
    # 2. 데이터 분할
    X = initial_data.drop('target', axis=1)
    y = initial_data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # 3. 적응형 특성 선택기 초기화
    logger.info("적응형 특성 선택기 초기화")
    
    feature_selector = AdaptiveFeatureSelector(
        config={
            'base_features': ['feature_0', 'feature_1'],  # 항상 포함할 기본 특성
            'selection_method': 'random_forest',
            'importance_threshold': 0.02,
            'min_features': 5,
            'max_features': 20,
            'use_validation': True,
            'track_importance_history': True,
            'auto_adjust_threshold': True,
            'auto_adjust_interval': 3
        }
    )
    
    # 4. 초기 특성 선택
    logger.info("\n초기 특성 선택")
    
    selected_features = feature_selector.select_features(
        X_train, y_train,
        validation_data=(X_val, y_val)
    )
    
    logger.info(f"선택된 특성 ({len(selected_features)}개): {selected_features}")
    
    # 5. 선택된 특성으로 모델 평가
    logger.info("\n선택된 특성으로 모델 평가")
    
    # 전체 특성 사용 평가
    all_features_result = evaluate_model(X_train, y_train, X_test, y_test)
    logger.info(f"전체 특성 사용 ({n_total_features}개):")
    logger.info(f"  RMSE: {all_features_result['rmse']:.6f}")
    logger.info(f"  MAE: {all_features_result['mae']:.6f}")
    logger.info(f"  R²: {all_features_result['r2']:.6f}")
    
    # 선택된 특성 사용 평가
    selected_features_result = evaluate_model(
        X_train, y_train, X_test, y_test, selected_features
    )
    logger.info(f"선택된 특성 사용 ({len(selected_features)}개):")
    logger.info(f"  RMSE: {selected_features_result['rmse']:.6f}")
    logger.info(f"  MAE: {selected_features_result['mae']:.6f}")
    logger.info(f"  R²: {selected_features_result['r2']:.6f}")
    
    # 6. 중요도 시각화
    logger.info("\n특성 중요도 시각화")
    feature_selector.plot_feature_importances(top_n=10, show_plot=False,
                                          save_path="logs/feature_importances.png")
    logger.info("특성 중요도 시각화 저장: logs/feature_importances.png")
    
    # 7. 시간에 따른 특성 중요도 변화 시뮬레이션
    logger.info("\n시간에 따른 특성 중요도 변화 시뮬레이션")
    
    # 결과 저장을 위한 데이터프레임
    results_df = pd.DataFrame()
    
    # 초기 결과 저장
    results_df = results_df.append({
        'timestep': 0,
        'all_features_rmse': all_features_result['rmse'],
        'all_features_r2': all_features_result['r2'],
        'selected_features_rmse': selected_features_result['rmse'],
        'selected_features_r2': selected_features_result['r2'],
        'n_selected_features': len(selected_features)
    }, ignore_index=True)
    
    # 시간 경과 시뮬레이션 (5개 타임스텝)
    for t in range(1, 6):
        logger.info(f"\n타임스텝 {t}:")
        
        # 중요 특성 변화 시뮬레이션
        new_important_features = [
            f'feature_{15 + t}',  # 시간에 따라 중요해지는 새로운 특성
            f'feature_{20 + t}',
            f'feature_{25 + t}'
        ]
        
        # 데이터 변화 시뮬레이션
        changed_data = simulate_feature_importance_change(
            initial_data,
            new_important_features,
            increase_factor=1.5 + t*0.5,
            random_state=42+t
        )
        
        # 노이즈 추가
        changed_data = add_noise_to_features(
            changed_data,
            noise_level=0.05 * t,
            random_state=42+t
        )
        
        # 데이터 분할
        X_new = changed_data.drop('target', axis=1)
        y_new = changed_data['target']
        
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
            X_new, y_new, test_size=0.2, random_state=42
        )
        
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
            X_train_new, y_train_new, test_size=0.25, random_state=42
        )
        
        # 적응형 특성 선택
        new_selected_features = feature_selector.select_features(
            X_train_new, y_train_new,
            validation_data=(X_val_new, y_val_new)
        )
        
        logger.info(f"새롭게 선택된 특성 ({len(new_selected_features)}개): {new_selected_features}")
        
        # 새로운 중요 특성이 선택되었는지 확인
        selected_new_important = [f for f in new_important_features if f in new_selected_features]
        logger.info(f"새로운 중요 특성 중 선택된 특성: {selected_new_important} "
                   f"({len(selected_new_important)}/{len(new_important_features)})")
        
        # 전체 특성 사용 평가
        all_features_result = evaluate_model(
            X_train_new, y_train_new, X_test_new, y_test_new
        )
        
        # 선택된 특성 사용 평가
        selected_features_result = evaluate_model(
            X_train_new, y_train_new, X_test_new, y_test_new, new_selected_features
        )
        
        logger.info(f"전체 특성 성능: RMSE={all_features_result['rmse']:.6f}, R²={all_features_result['r2']:.6f}")
        logger.info(f"선택 특성 성능: RMSE={selected_features_result['rmse']:.6f}, R²={selected_features_result['r2']:.6f}")
        
        # 결과 저장
        results_df = results_df.append({
            'timestep': t,
            'all_features_rmse': all_features_result['rmse'],
            'all_features_r2': all_features_result['r2'],
            'selected_features_rmse': selected_features_result['rmse'],
            'selected_features_r2': selected_features_result['r2'],
            'n_selected_features': len(new_selected_features)
        }, ignore_index=True)
    
    # 8. 특성 중요도 변화 추이 시각화
    logger.info("\n특성 중요도 변화 추이 시각화")
    feature_selector.plot_importance_trend(top_n=8, show_plot=False,
                                       save_path="logs/feature_importance_trend.png")
    logger.info("특성 중요도 변화 추이 시각화 저장: logs/feature_importance_trend.png")
    
    # 9. 성능 변화 결과 시각화
    logger.info("\n성능 변화 결과 시각화")
    
    plt.figure(figsize=(12, 10))
    
    # RMSE 변화 시각화
    plt.subplot(2, 1, 1)
    plt.plot(results_df['timestep'], results_df['all_features_rmse'], 'o-', 
            label='전체 특성')
    plt.plot(results_df['timestep'], results_df['selected_features_rmse'], 's-', 
            label='선택된 특성')
    plt.title('시간에 따른 RMSE 변화')
    plt.xlabel('타임스텝')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # R² 변화 시각화
    plt.subplot(2, 1, 2)
    plt.plot(results_df['timestep'], results_df['all_features_r2'], 'o-', 
            label='전체 특성')
    plt.plot(results_df['timestep'], results_df['selected_features_r2'], 's-', 
            label='선택된 특성')
    plt.title('시간에 따른 R² 변화')
    plt.xlabel('타임스텝')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("logs/performance_comparison.png")
    logger.info("성능 변화 시각화 저장: logs/performance_comparison.png")
    
    # 10. 선택된 특성 수 변화 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['timestep'], results_df['n_selected_features'], 'o-')
    plt.title('시간에 따른 선택된 특성 수 변화')
    plt.xlabel('타임스텝')
    plt.ylabel('선택된 특성 수')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("logs/selected_features_count.png")
    logger.info("선택된 특성 수 변화 시각화 저장: logs/selected_features_count.png")
    
    # 11. 모델 저장 및 로드 테스트
    logger.info("\n모델 저장 및 로드 테스트")
    
    save_path = "logs/feature_selector.joblib"
    feature_selector.save(save_path)
    
    loaded_selector = AdaptiveFeatureSelector.load(save_path)
    logger.info(f"모델 로드 완료, 선택 횟수: {loaded_selector.selection_count}")
    
    logger.info("적응형 특성 선택 메커니즘 예제 완료")

if __name__ == "__main__":
    # 로그 디렉토리 생성
    os.makedirs("logs", exist_ok=True)
    
    run_adaptive_feature_selection_example() 