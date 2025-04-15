import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass

from ..forecasting.time_series_predictor import TimeSeriesPredictor
from ..forecasting.model_factory import TimeSeriesPredictorFactory, ModelType

@dataclass
class EnsembleWeight:
    model_name: str
    weight: float

class TimeSeriesEnsemble:
    """시계열 예측을 위한 앙상블 모델"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        초기화
        
        Args:
            config: 설정 파라미터
        """
        self.config = config or {
            'ensemble_method': 'weighted',  # 'weighted', 'average', 'median', 'optimal'
            'optimization_metric': 'rmse',  # 'rmse', 'mae', 'mape'
            'model_types': [ModelType.LSTM, ModelType.GRU, ModelType.BIDIRECTIONAL_LSTM],
            'model_configs': None,
            'sequence_length': 60,
            'n_features': 1
        }
        self.models = []
        self.weights = []
        self.is_fitted = False
        
    def build_models(self) -> None:
        """앙상블을 구성하는 모델들 생성"""
        model_types = self.config.get('model_types', [ModelType.LSTM])
        model_configs = self.config.get('model_configs', None)
        
        # 모델 설정이 없으면 동일한 기본 설정으로 초기화
        if model_configs is None:
            model_configs = [None] * len(model_types)
        
        # 모델 생성
        for i, model_type in enumerate(model_types):
            config = model_configs[i] or {
                'sequence_length': self.config.get('sequence_length', 60),
                'n_features': self.config.get('n_features', 1)
            }
            model = TimeSeriesPredictorFactory.create_model(model_type, config)
            self.models.append(model)
            
        # 초기 가중치 설정 (동일 가중치)
        n_models = len(self.models)
        self.weights = [1.0 / n_models] * n_models
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              validation_data: Tuple[np.ndarray, np.ndarray],
              epochs: int = 100, batch_size: int = 32, patience: int = 10) -> None:
        """
        모든 모델 훈련
        
        Args:
            X_train: 훈련 데이터 특성
            y_train: 훈련 데이터 타겟
            validation_data: 검증 데이터 (X_val, y_val)
            epochs: 에포크 수
            batch_size: 배치 크기
            patience: 조기 종료 인내
        """
        if not self.models:
            self.build_models()
        
        # 모든 모델 훈련
        for i, model in enumerate(self.models):
            print(f"모델 {i+1}/{len(self.models)} 훈련 중...")
            model.train(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
        
        # 검증 데이터로 최적 가중치 계산
        X_val, y_val = validation_data
        ensemble_method = self.config.get('ensemble_method', 'weighted')
        
        if ensemble_method == 'optimal':
            self._optimize_weights(X_val, y_val)
        
        self.is_fitted = True
        
    def _optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        검증 데이터를 사용하여 최적 가중치 계산
        
        Args:
            X_val: 검증 데이터 특성
            y_val: 검증 데이터 타겟
        """
        # 각 모델의 예측 생성
        predictions = []
        for model in self.models:
            pred = model.predict(X_val)
            predictions.append(pred)
        
        # 목적 함수: 가중 평균 예측의 오차 최소화
        def objective(weights):
            # 가중치 합이 1이 되도록 정규화
            weights = weights / np.sum(weights)
            
            # 가중 평균 예측 계산
            weighted_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                weighted_pred += weights[i] * pred
            
            # 오차 계산
            metric = self.config.get('optimization_metric', 'rmse')
            if metric == 'rmse':
                error = np.sqrt(np.mean((weighted_pred - y_val) ** 2))
            elif metric == 'mae':
                error = np.mean(np.abs(weighted_pred - y_val))
            elif metric == 'mape':
                error = np.mean(np.abs((y_val - weighted_pred) / y_val)) * 100
            else:
                error = np.sqrt(np.mean((weighted_pred - y_val) ** 2))  # 기본값은 RMSE
                
            return error
        
        # 초기 가중치
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # 제약 조건: 가중치 합이 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # 각 가중치는 0 이상 1 이하
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        # 최적화
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraints
        )
        
        # 최적 가중치 적용
        self.weights = result.x / np.sum(result.x)  # 정규화
        
        print("최적 가중치 계산 완료:")
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            model_type = self.config.get('model_types', ['Unknown'])[i]
            print(f"  모델 {i+1} ({model_type}): {weight:.4f}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        앙상블 예측 수행
        
        Args:
            X: 예측할 데이터 특성
            
        Returns:
            예측 결과
        """
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        ensemble_method = self.config.get('ensemble_method', 'weighted')
        predictions = []
        
        # 각 모델의 예측 생성
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # 앙상블 방식에 따라 예측 결합
        if ensemble_method == 'average':
            # 단순 평균
            final_pred = np.mean(predictions, axis=0)
        elif ensemble_method == 'median':
            # 중앙값
            final_pred = np.median(predictions, axis=0)
        elif ensemble_method == 'weighted' or ensemble_method == 'optimal':
            # 가중 평균
            final_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                final_pred += self.weights[i] * pred
        else:
            raise ValueError(f"지원하지 않는 앙상블 방법: {ensemble_method}")
        
        return final_pred
        
    def forecast(self, last_sequence: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        미래 n 기간 예측
        
        Args:
            last_sequence: 마지막 시퀀스
            horizon: 예측 기간
            
        Returns:
            예측 결과
        """
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        ensemble_method = self.config.get('ensemble_method', 'weighted')
        forecasts = []
        
        # 각 모델의 예측 생성
        for model in self.models:
            pred = model.forecast(last_sequence, horizon)
            forecasts.append(pred)
        
        # 앙상블 방식에 따라 예측 결합
        if ensemble_method == 'average':
            # 단순 평균
            final_forecast = np.mean(forecasts, axis=0)
        elif ensemble_method == 'median':
            # 중앙값
            final_forecast = np.median(forecasts, axis=0)
        elif ensemble_method == 'weighted' or ensemble_method == 'optimal':
            # 가중 평균
            final_forecast = np.zeros_like(forecasts[0])
            for i, pred in enumerate(forecasts):
                final_forecast += self.weights[i] * pred
        else:
            raise ValueError(f"지원하지 않는 앙상블 방법: {ensemble_method}")
        
        return final_forecast
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        앙상블 모델 평가
        
        Args:
            X_test: 테스트 데이터 특성
            y_test: 테스트 데이터 타겟
            
        Returns:
            평가 지표
        """
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 앙상블 예측
        y_pred = self.predict(X_test)
        
        # 원래 스케일로 변환 (첫 번째 모델의 스케일러 사용)
        y_test_orig = self.models[0].scaler.inverse_transform(y_test)
        y_pred_orig = self.models[0].scaler.inverse_transform(y_pred)
        
        # 개별 모델 평가
        individual_metrics = []
        for i, model in enumerate(self.models):
            model_pred = model.predict(X_test)
            model_pred_orig = model.scaler.inverse_transform(model_pred)
            
            # RMSE 계산
            rmse = np.sqrt(np.mean((model_pred_orig - y_test_orig) ** 2))
            mae = np.mean(np.abs(model_pred_orig - y_test_orig))
            
            individual_metrics.append({
                'model_index': i,
                'model_type': self.config.get('model_types', ['Unknown'])[i],
                'weight': self.weights[i],
                'rmse': float(rmse),
                'mae': float(mae)
            })
        
        # 앙상블 모델 평가 지표
        metrics = {
            'rmse': float(np.sqrt(np.mean((y_pred_orig - y_test_orig) ** 2))),
            'mae': float(np.mean(np.abs(y_pred_orig - y_test_orig))),
            'r2': float(1 - np.sum((y_test_orig - y_pred_orig) ** 2) / np.sum((y_test_orig - np.mean(y_test_orig)) ** 2)),
            'individual_metrics': individual_metrics
        }
        
        return metrics
    
    def plot_predictions(self, X_test: np.ndarray, y_test: np.ndarray, title: str = '앙상블 예측 결과') -> None:
        """
        앙상블 예측 결과 시각화
        
        Args:
            X_test: 테스트 데이터 특성
            y_test: 테스트 데이터 타겟
            title: 그래프 제목
        """
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 모든 모델 및 앙상블 예측
        ensemble_pred = self.predict(X_test)
        
        # 원래 스케일로 변환
        y_test_orig = self.models[0].scaler.inverse_transform(y_test)
        ensemble_pred_orig = self.models[0].scaler.inverse_transform(ensemble_pred)
        
        # 개별 모델 예측
        model_preds = []
        for model in self.models:
            pred = model.predict(X_test)
            pred_orig = model.scaler.inverse_transform(pred)
            model_preds.append(pred_orig)
        
        # 시각화
        plt.figure(figsize=(15, 8))
        
        # 실제 값
        plt.plot(y_test_orig, 'k-', label='실제 값', linewidth=2)
        
        # 개별 모델 예측 (더 연한 색상)
        for i, pred in enumerate(model_preds):
            plt.plot(pred, '--', alpha=0.5, label=f'모델 {i+1}')
        
        # 앙상블 예측 (강조)
        plt.plot(ensemble_pred_orig, 'r-', label='앙상블 예측', linewidth=2)
        
        plt.title(title)
        plt.xlabel('시간')
        plt.ylabel('값')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def get_ensemble_weights(self) -> List[EnsembleWeight]:
        """
        앙상블 가중치 반환
        
        Returns:
            가중치 목록
        """
        weights = []
        for i, w in enumerate(self.weights):
            model_type = self.config.get('model_types', ['Unknown'])[i]
            weights.append(EnsembleWeight(
                model_name=f"{model_type}",
                weight=float(w)
            ))
        return weights 