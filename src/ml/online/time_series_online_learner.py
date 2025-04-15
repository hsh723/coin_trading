#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
시계열 모델을 위한 온라인 학습 시스템
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import joblib
import json
import time
import logging
from dataclasses import dataclass

from ..forecasting.time_series_predictor import TimeSeriesPredictor
from ..ensemble.time_series_ensemble import TimeSeriesEnsemble
from ..forecasting.model_factory import TimeSeriesPredictorFactory, ModelType

@dataclass
class OnlineLearningResult:
    """온라인 학습 결과"""
    model_id: str
    update_time: str
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    performance_change: Dict[str, float]
    data_points_used: int

class OnlineLearner:
    """시계열 예측을 위한 온라인 학습 시스템"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        초기화
        
        Args:
            config: 설정 파라미터
        """
        self.config = config or {
            'model_type': 'ensemble',  # 'single' 또는 'ensemble'
            'model_path': None,  # 기존 모델 경로 (없으면 새로 생성)
            'update_frequency': 24,  # 업데이트 주기 (시간)
            'min_data_points': 48,  # 학습에 필요한 최소 데이터 포인트
            'max_data_points': 1000,  # 학습에 사용할 최대 데이터 포인트
            'min_performance_gain': 0.01,  # 최소 성능 향상 요구치 (RMSE 감소율)
            'validation_size': 0.3,  # 검증 데이터 비율
            'sequence_length': 48,  # 시퀀스 길이
            'n_features': 1,  # 특성 수
            'target_column': 'close',  # 타겟 열
            'ensemble_config': None,  # 앙상블 모델 설정
            'single_model_config': None,  # 단일 모델 설정
            'model_save_path': 'models/online/',  # 모델 저장 경로
            'rolling_window': True,  # 롤링 윈도우 방식 사용 (가장 오래된 데이터 제거)
            'fine_tune_epochs': 10,  # 미세 조정 에포크 수
            'full_retrain_threshold': 0.05,  # 전체 재학습 임계값 (RMSE 증가율)
        }
        
        self.model = None
        self.last_update_time = None
        self.update_history = []
        self.model_id = None
        self.logger = logging.getLogger(__name__)
        
        # 모델 ID 생성
        self._generate_model_id()
        
        # 모델 초기화
        self._initialize_model()
    
    def _generate_model_id(self) -> None:
        """모델 ID 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.config.get('model_type', 'unknown')
        self.model_id = f"{model_type}_{timestamp}"
    
    def _initialize_model(self) -> None:
        """모델 초기화"""
        model_path = self.config.get('model_path')
        
        if model_path and os.path.exists(model_path):
            # 기존 모델 로드
            self.load_model(model_path)
        else:
            # 새 모델 생성
            if self.config.get('model_type') == 'ensemble':
                # 앙상블 모델 생성
                ensemble_config = self.config.get('ensemble_config') or {
                    'sequence_length': self.config.get('sequence_length', 48),
                    'n_features': self.config.get('n_features', 1),
                    'model_types': [ModelType.LSTM, ModelType.GRU, ModelType.BIDIRECTIONAL_LSTM],
                    'ensemble_method': 'weighted'
                }
                self.model = TimeSeriesEnsemble(ensemble_config)
                self.model.build_models()
            else:
                # 단일 모델 생성
                single_model_config = self.config.get('single_model_config') or {
                    'sequence_length': self.config.get('sequence_length', 48),
                    'n_features': self.config.get('n_features', 1)
                }
                model_type = single_model_config.get('model_type', ModelType.LSTM)
                self.model = TimeSeriesPredictorFactory.create_model(model_type, single_model_config)
    
    def _should_update(self) -> bool:
        """모델 업데이트 필요 여부 확인"""
        if self.last_update_time is None:
            return True
        
        update_frequency = self.config.get('update_frequency', 24)  # 시간 단위
        time_since_update = datetime.now() - self.last_update_time
        return time_since_update.total_seconds() / 3600 >= update_frequency
    
    def update(self, new_data: pd.DataFrame, features: List[str] = None) -> Optional[OnlineLearningResult]:
        """
        새 데이터로 모델 업데이트
        
        Args:
            new_data: 새 데이터
            features: 사용할 특성 목록 (None이면 모든 열 사용)
            
        Returns:
            업데이트 결과
        """
        if not self._should_update():
            self.logger.info("아직 업데이트 시간이 아닙니다.")
            return None
        
        # 특성 목록 설정
        if features is None:
            features = list(new_data.columns)
        
        # 필요한 특성 확인
        target_col = self.config.get('target_column', 'close')
        if target_col not in features:
            features.append(target_col)
        
        # 데이터 준비
        min_data_points = self.config.get('min_data_points', 48)
        max_data_points = self.config.get('max_data_points', 1000)
        
        if len(new_data) < min_data_points:
            self.logger.warning(f"데이터가 부족합니다. 최소 {min_data_points}개 필요, 현재 {len(new_data)}개.")
            return None
        
        # 데이터 크기 제한
        if self.config.get('rolling_window', True) and len(new_data) > max_data_points:
            new_data = new_data.iloc[-max_data_points:]
        
        # 업데이트 방법 결정
        # 1. 첫 학습이면 전체 훈련
        # 2. 아니면 검증 데이터로 성능 체크 후 결정
        
        if not hasattr(self.model, 'is_fitted') or not self.model.is_fitted:
            # 첫 학습 - 전체 훈련
            update_result = self._full_train(new_data, features, target_col)
        else:
            # 검증 성능 체크 후 업데이트 방법 결정
            validation_size = self.config.get('validation_size', 0.3)
            val_size = int(len(new_data) * validation_size)
            validation_data = new_data.iloc[-val_size:].copy()
            
            # 성능 체크
            metrics_before = self._evaluate_on_data(validation_data, features, target_col)
            
            # 미세 조정
            update_result = self._fine_tune(new_data, features, target_col, metrics_before)
            
            # 성능이 크게 저하되면 전체 재훈련
            full_retrain_threshold = self.config.get('full_retrain_threshold', 0.05)
            
            if update_result and update_result.performance_change.get('rmse_change', 0) > full_retrain_threshold:
                self.logger.info(f"성능이 크게 저하되어 전체 재훈련 수행 (RMSE 증가: {update_result.performance_change['rmse_change']:.4f})")
                update_result = self._full_train(new_data, features, target_col)
        
        # 업데이트 시간 기록
        self.last_update_time = datetime.now()
        
        # 모델 저장
        if update_result:
            self._save_model()
            self.update_history.append(update_result)
        
        return update_result
    
    def _evaluate_on_data(self, data: pd.DataFrame, features: List[str], target_col: str) -> Dict[str, float]:
        """
        데이터에 대한 모델 성능 평가
        
        Args:
            data: 평가할 데이터
            features: 특성 목록
            target_col: 타겟 열
            
        Returns:
            성능 지표
        """
        # 데이터 선택
        data = data[features].copy()
        
        # 앙상블 모델인지 단일 모델인지 확인
        is_ensemble = isinstance(self.model, TimeSeriesEnsemble)
        
        # 데이터 준비
        sequence_length = self.config.get('sequence_length', 48)
        
        if is_ensemble:
            # 앙상블 모델용 데이터 준비
            X, y = self.model.models[0].prepare_multivariate_data(data, target_col)
        else:
            # 단일 모델용 데이터 준비
            if len(features) > 1:
                X, y = self.model.prepare_multivariate_data(data, target_col)
            else:
                X, y = self.model.prepare_data(data, target_col)
        
        # 평가
        if is_ensemble:
            metrics = self.model.evaluate(X, y)
            # 필요한 지표만 추출
            return {
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics.get('r2', 0)
            }
        else:
            metrics = self.model.evaluate(X, y)
            return metrics
    
    def _fine_tune(self, data: pd.DataFrame, features: List[str], target_col: str, 
                  metrics_before: Dict[str, float]) -> Optional[OnlineLearningResult]:
        """
        모델 미세 조정
        
        Args:
            data: 학습 데이터
            features: 특성 목록
            target_col: 타겟 열
            metrics_before: 업데이트 전 성능 지표
            
        Returns:
            업데이트 결과
        """
        self.logger.info("모델 미세 조정 시작")
        start_time = time.time()
        
        # 데이터 분할
        validation_size = self.config.get('validation_size', 0.3)
        val_size = int(len(data) * validation_size)
        train_data = data.iloc[:-val_size].copy()
        val_data = data.iloc[-val_size:].copy()
        
        # 데이터 선택
        train_data = train_data[features].copy()
        val_data = val_data[features].copy()
        
        # 앙상블 모델인지 단일 모델인지 확인
        is_ensemble = isinstance(self.model, TimeSeriesEnsemble)
        
        # 데이터 준비
        if is_ensemble:
            # 앙상블 모델용 데이터 준비
            X_train, y_train = self.model.models[0].prepare_multivariate_data(train_data, target_col)
            X_val, y_val = self.model.models[0].prepare_multivariate_data(val_data, target_col)
        else:
            # 단일 모델용 데이터 준비
            if len(features) > 1:
                X_train, y_train = self.model.prepare_multivariate_data(train_data, target_col)
                X_val, y_val = self.model.prepare_multivariate_data(val_data, target_col)
            else:
                X_train, y_train = self.model.prepare_data(train_data, target_col)
                X_val, y_val = self.model.prepare_data(val_data, target_col)
        
        # 미세 조정
        fine_tune_epochs = self.config.get('fine_tune_epochs', 10)
        
        if is_ensemble:
            # 앙상블 모델 미세 조정
            for i, model in enumerate(self.model.models):
                self.logger.info(f"앙상블 모델 {i+1}/{len(self.model.models)} 미세 조정 중...")
                model.train(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=fine_tune_epochs,
                    batch_size=32,
                    patience=5
                )
        else:
            # 단일 모델 미세 조정
            self.model.train(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=fine_tune_epochs,
                batch_size=32,
                patience=5
            )
        
        # 미세 조정 후 성능 평가
        metrics_after = self._evaluate_on_data(val_data, features, target_col)
        
        # 성능 변화 계산
        performance_change = {
            'rmse_change': (metrics_after['rmse'] - metrics_before['rmse']) / metrics_before['rmse'],
            'mae_change': (metrics_after['mae'] - metrics_before['mae']) / metrics_before['mae'],
            'r2_change': metrics_after.get('r2', 0) - metrics_before.get('r2', 0)
        }
        
        # 결과 로깅
        self.logger.info(f"미세 조정 완료: {time.time() - start_time:.2f}초 소요")
        self.logger.info(f"RMSE: {metrics_before['rmse']:.6f} -> {metrics_after['rmse']:.6f} (변화: {performance_change['rmse_change']*100:.2f}%)")
        self.logger.info(f"MAE: {metrics_before['mae']:.6f} -> {metrics_after['mae']:.6f} (변화: {performance_change['mae_change']*100:.2f}%)")
        
        # 최소 성능 향상 요구치 확인
        min_performance_gain = self.config.get('min_performance_gain', 0.01)
        
        # 성능이 개선되었거나 최소 성능 향상 요구치보다 적게 감소한 경우 모델 유지
        if performance_change['rmse_change'] <= 0 or abs(performance_change['rmse_change']) < min_performance_gain:
            return OnlineLearningResult(
                model_id=self.model_id,
                update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                performance_change=performance_change,
                data_points_used=len(train_data)
            )
        else:
            # 성능이 크게 저하된 경우 결과 반환 (호출자에서 전체 재훈련 결정)
            return OnlineLearningResult(
                model_id=self.model_id,
                update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                performance_change=performance_change,
                data_points_used=len(train_data)
            )
    
    def _full_train(self, data: pd.DataFrame, features: List[str], target_col: str) -> OnlineLearningResult:
        """
        모델 전체 훈련
        
        Args:
            data: 학습 데이터
            features: 특성 목록
            target_col: 타겟 열
            
        Returns:
            업데이트 결과
        """
        self.logger.info("모델 전체 훈련 시작")
        start_time = time.time()
        
        # 데이터 분할
        validation_size = self.config.get('validation_size', 0.3)
        val_size = int(len(data) * validation_size)
        train_data = data.iloc[:-val_size].copy()
        val_data = data.iloc[-val_size:].copy()
        
        # 데이터 선택
        train_data = train_data[features].copy()
        val_data = val_data[features].copy()
        
        # 앙상블 모델인지 단일 모델인지 확인
        is_ensemble = isinstance(self.model, TimeSeriesEnsemble)
        
        # 전체 훈련 전 성능 측정 (모델이 이미 학습된 경우만)
        metrics_before = None
        if hasattr(self.model, 'is_fitted') and self.model.is_fitted:
            try:
                metrics_before = self._evaluate_on_data(val_data, features, target_col)
            except Exception as e:
                self.logger.warning(f"훈련 전 성능 측정 실패: {e}")
                metrics_before = {'rmse': float('inf'), 'mae': float('inf'), 'r2': 0}
        else:
            metrics_before = {'rmse': float('inf'), 'mae': float('inf'), 'r2': 0}
        
        # 데이터 준비
        if is_ensemble:
            # 앙상블 모델용 데이터 준비
            X_train, y_train = self.model.models[0].prepare_multivariate_data(train_data, target_col)
            X_val, y_val = self.model.models[0].prepare_multivariate_data(val_data, target_col)
            
            # 앙상블 모델 전체 훈련
            self.model.train(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                patience=10
            )
        else:
            # 단일 모델용 데이터 준비
            if len(features) > 1:
                X_train, y_train = self.model.prepare_multivariate_data(train_data, target_col)
                X_val, y_val = self.model.prepare_multivariate_data(val_data, target_col)
            else:
                X_train, y_train = self.model.prepare_data(train_data, target_col)
                X_val, y_val = self.model.prepare_data(val_data, target_col)
            
            # 단일 모델 전체 훈련
            self.model.train(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                patience=10
            )
        
        # 훈련 후 성능 평가
        metrics_after = self._evaluate_on_data(val_data, features, target_col)
        
        # 성능 변화 계산
        performance_change = {
            'rmse_change': (metrics_after['rmse'] - metrics_before['rmse']) / max(metrics_before['rmse'], 1e-10),
            'mae_change': (metrics_after['mae'] - metrics_before['mae']) / max(metrics_before['mae'], 1e-10),
            'r2_change': metrics_after.get('r2', 0) - metrics_before.get('r2', 0)
        }
        
        # 결과 로깅
        self.logger.info(f"전체 훈련 완료: {time.time() - start_time:.2f}초 소요")
        self.logger.info(f"RMSE: {metrics_before['rmse']:.6f} -> {metrics_after['rmse']:.6f} (변화: {performance_change['rmse_change']*100:.2f}%)")
        self.logger.info(f"MAE: {metrics_before['mae']:.6f} -> {metrics_after['mae']:.6f} (변화: {performance_change['mae_change']*100:.2f}%)")
        
        return OnlineLearningResult(
            model_id=self.model_id,
            update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            performance_change=performance_change,
            data_points_used=len(train_data)
        )
    
    def predict(self, data: pd.DataFrame, features: List[str] = None) -> np.ndarray:
        """
        예측 수행
        
        Args:
            data: 예측할 데이터
            features: 사용할 특성 목록
            
        Returns:
            예측 결과
        """
        if not hasattr(self.model, 'is_fitted') or not self.model.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        if features is None:
            features = list(data.columns)
        
        # 앙상블 모델인지 단일 모델인지 확인
        is_ensemble = isinstance(self.model, TimeSeriesEnsemble)
        
        # 입력 데이터 준비
        sequence_length = self.config.get('sequence_length', 48)
        
        if len(data) < sequence_length:
            raise ValueError(f"데이터가 부족합니다. 최소 {sequence_length}개 필요, 현재 {len(data)}개.")
        
        # 예측에 사용할 데이터 선택
        data = data[features].copy()
        
        # 입력 데이터 준비
        if is_ensemble:
            # 앙상블 모델용 데이터 준비
            X, _ = self.model.models[0].prepare_multivariate_data(
                data.iloc[-sequence_length:],
                target_col=self.config.get('target_column', 'close')
            )
            # 데이터 형태 확인
            if len(X) == 0:
                raise ValueError("입력 데이터 준비 실패")
            
            # 앙상블 모델 예측
            predictions = self.model.predict(np.array([X[0]]))
        else:
            # 단일 모델용 데이터 준비
            if len(features) > 1:
                X, _ = self.model.prepare_multivariate_data(
                    data.iloc[-sequence_length:],
                    target_col=self.config.get('target_column', 'close')
                )
            else:
                X, _ = self.model.prepare_data(
                    data.iloc[-sequence_length:],
                    target_col=self.config.get('target_column', 'close')
                )
            
            # 데이터 형태 확인
            if len(X) == 0:
                raise ValueError("입력 데이터 준비 실패")
            
            # 단일 모델 예측
            predictions = self.model.predict(np.array([X[0]]))
        
        return predictions
    
    def forecast(self, data: pd.DataFrame, horizon: int = 1, 
                features: List[str] = None) -> np.ndarray:
        """
        미래 예측 수행
        
        Args:
            data: 마지막 시퀀스 데이터
            horizon: 예측 기간
            features: 사용할 특성 목록
            
        Returns:
            예측 결과
        """
        if not hasattr(self.model, 'is_fitted') or not self.model.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        if features is None:
            features = list(data.columns)
        
        # 앙상블 모델인지 단일 모델인지 확인
        is_ensemble = isinstance(self.model, TimeSeriesEnsemble)
        
        # 입력 데이터 준비
        sequence_length = self.config.get('sequence_length', 48)
        
        if len(data) < sequence_length:
            raise ValueError(f"데이터가 부족합니다. 최소 {sequence_length}개 필요, 현재 {len(data)}개.")
        
        # 예측에 사용할 데이터 선택
        data = data[features].iloc[-sequence_length:].copy()
        
        # 미래 예측
        if is_ensemble:
            # 앙상블 모델 예측
            last_sequence = data.values
            forecasts = self.model.forecast(last_sequence, horizon)
        else:
            # 단일 모델 예측
            last_sequence = data.values.reshape(sequence_length, len(features))
            forecasts = self.model.forecast(last_sequence, horizon)
        
        return forecasts
    
    def _save_model(self) -> None:
        """모델 저장"""
        model_save_path = self.config.get('model_save_path', 'models/online/')
        os.makedirs(model_save_path, exist_ok=True)
        
        # 앙상블 모델인지 단일 모델인지 확인
        is_ensemble = isinstance(self.model, TimeSeriesEnsemble)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = 'ensemble' if is_ensemble else 'single'
        
        # 메인 모델 저장
        if is_ensemble:
            # 앙상블 모델 저장
            for i, model in enumerate(self.model.models):
                model_path = os.path.join(model_save_path, f"{self.model_id}_model_{i}.h5")
                scaler_path = os.path.join(model_save_path, f"{self.model_id}_scaler_{i}.pkl")
                model.save(model_path, scaler_path)
            
            # 앙상블 가중치 저장
            weights_path = os.path.join(model_save_path, f"{self.model_id}_weights.json")
            weights = self.model.weights
            with open(weights_path, 'w') as f:
                json.dump({
                    'weights': weights,
                    'model_types': [str(mt) for mt in self.model.config.get('model_types', [])]
                }, f)
        else:
            # 단일 모델 저장
            model_path = os.path.join(model_save_path, f"{self.model_id}_model.h5")
            scaler_path = os.path.join(model_save_path, f"{self.model_id}_scaler.pkl")
            self.model.save(model_path, scaler_path)
        
        # 설정 저장
        config_path = os.path.join(model_save_path, f"{self.model_id}_config.json")
        config_copy = self.config.copy()
        
        # ModelType Enum은 JSON으로 직렬화할 수 없으므로 문자열로 변환
        if 'model_types' in config_copy.get('ensemble_config', {}):
            config_copy['ensemble_config']['model_types'] = [str(mt) for mt in config_copy['ensemble_config']['model_types']]
        
        with open(config_path, 'w') as f:
            json.dump(config_copy, f, indent=4)
        
        # 업데이트 기록 저장
        history_path = os.path.join(model_save_path, f"{self.model_id}_history.json")
        with open(history_path, 'w') as f:
            history_data = []
            for result in self.update_history:
                history_data.append({
                    'update_time': result.update_time,
                    'metrics_before': result.metrics_before,
                    'metrics_after': result.metrics_after,
                    'performance_change': result.performance_change,
                    'data_points_used': result.data_points_used
                })
            json.dump(history_data, f, indent=4)
        
        self.logger.info(f"모델 저장 완료: {model_save_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        저장된 모델 로드
        
        Args:
            model_path: 모델 디렉토리 또는 설정 파일 경로
        """
        # 설정 파일인지 디렉토리인지 확인
        if model_path.endswith('_config.json'):
            config_path = model_path
            model_dir = os.path.dirname(model_path)
            model_id = os.path.basename(config_path).replace('_config.json', '')
        else:
            model_dir = model_path
            # 가장 최근 설정 파일 찾기
            config_files = [f for f in os.listdir(model_dir) if f.endswith('_config.json')]
            if not config_files:
                raise FileNotFoundError(f"모델 디렉토리에 설정 파일이 없습니다: {model_dir}")
            
            config_files.sort(reverse=True)  # 가장 최근 파일 우선
            config_path = os.path.join(model_dir, config_files[0])
            model_id = config_files[0].replace('_config.json', '')
        
        # 설정 파일 로드
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 모델 ID 설정
        self.model_id = model_id
        
        # 앙상블 모델인지 단일 모델인지 확인
        model_type = self.config.get('model_type', 'single')
        
        if model_type == 'ensemble':
            # 앙상블 모델 로드
            # 앙상블 설정
            ensemble_config = self.config.get('ensemble_config', {})
            
            # ModelType Enum 문자열 변환
            if 'model_types' in ensemble_config:
                model_types = []
                for mt_str in ensemble_config['model_types']:
                    for mt in ModelType:
                        if str(mt) == mt_str or mt.name == mt_str:
                            model_types.append(mt)
                            break
                ensemble_config['model_types'] = model_types
            
            # 앙상블 모델 생성
            self.model = TimeSeriesEnsemble(ensemble_config)
            self.model.build_models()
            
            # 개별 모델 로드
            for i, model in enumerate(self.model.models):
                model_file = os.path.join(model_dir, f"{model_id}_model_{i}.h5")
                scaler_file = os.path.join(model_dir, f"{model_id}_scaler_{i}.pkl")
                
                if os.path.exists(model_file) and os.path.exists(scaler_file):
                    sequence_length = ensemble_config.get('sequence_length', 48)
                    n_features = ensemble_config.get('n_features', 1)
                    
                    self.model.models[i] = TimeSeriesPredictor.load(
                        model_file, scaler_file, 
                        sequence_length=sequence_length, 
                        n_features=n_features
                    )
                    self.model.is_fitted = True
                else:
                    self.logger.warning(f"모델 파일이 없습니다: {model_file} 또는 {scaler_file}")
            
            # 가중치 로드
            weights_file = os.path.join(model_dir, f"{model_id}_weights.json")
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as f:
                    weights_data = json.load(f)
                    self.model.weights = weights_data.get('weights', [1.0/len(self.model.models)] * len(self.model.models))
            else:
                self.logger.warning(f"가중치 파일이 없습니다: {weights_file}")
                self.model.weights = [1.0/len(self.model.models)] * len(self.model.models)
        else:
            # 단일 모델 로드
            model_file = os.path.join(model_dir, f"{model_id}_model.h5")
            scaler_file = os.path.join(model_dir, f"{model_id}_scaler.pkl")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                sequence_length = self.config.get('sequence_length', 48)
                n_features = self.config.get('n_features', 1)
                
                self.model = TimeSeriesPredictor.load(
                    model_file, scaler_file, 
                    sequence_length=sequence_length, 
                    n_features=n_features
                )
            else:
                self.logger.warning(f"모델 파일이 없습니다: {model_file} 또는 {scaler_file}")
                # 새 모델 생성
                self._initialize_model()
        
        # 업데이트 기록 로드
        history_file = os.path.join(model_dir, f"{model_id}_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history_data = json.load(f)
                self.update_history = []
                for item in history_data:
                    self.update_history.append(OnlineLearningResult(
                        model_id=self.model_id,
                        update_time=item['update_time'],
                        metrics_before=item['metrics_before'],
                        metrics_after=item['metrics_after'],
                        performance_change=item['performance_change'],
                        data_points_used=item['data_points_used']
                    ))
            
            # 마지막 업데이트 시간 설정
            if self.update_history:
                last_update = self.update_history[-1].update_time
                self.last_update_time = datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S")
        
        self.logger.info(f"모델 로드 완료: {model_id}")
        
    def get_update_history(self) -> List[OnlineLearningResult]:
        """업데이트 기록 반환"""
        return self.update_history 