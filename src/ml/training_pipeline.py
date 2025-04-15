import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
import optuna
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .forecasting.time_series_predictor import TimeSeriesPredictor
from .forecasting.model_factory import TimeSeriesPredictorFactory, ModelType

@dataclass
class TrainingResult:
    model: object
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    validation_predictions: Optional[pd.Series] = None
    test_predictions: Optional[pd.Series] = None
    best_params: Optional[Dict[str, Any]] = None
    training_history: Optional[Dict[str, List[float]]] = None

class ModelTrainingPipeline:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'n_splits': 5,
            'test_size': 0.2,
            'optimization_trials': 30,
            'sequence_length': 60,
            'target_column': 'close',
            'model_type': ModelType.LSTM,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
            'model_save_path': 'models/time_series/',
            'forecast_horizon': 1,
            'validation_size': 0.2
        }
        
    def run_pipeline(self, data: pd.DataFrame, 
                    features: Optional[List[str]] = None, 
                    target: Optional[str] = None) -> TrainingResult:
        """전체 훈련 파이프라인 실행"""
        # 설정값 및 기본값 설정
        target_col = target or self.config.get('target_column', 'close')
        sequence_length = self.config.get('sequence_length', 60)
        
        # 데이터 준비
        processed_data = self._preprocess_data(data, features)
        
        # 학습/검증/테스트 데이터 분할
        train_data, val_data, test_data = self._split_data(processed_data)
        
        # 모델 유형 선택
        model_type = self.config.get('model_type', ModelType.LSTM)
        
        # 하이퍼파라미터 최적화
        best_params = None
        if self.config.get('optimize_hyperparameters', True):
            best_params = self._optimize_hyperparameters(train_data, val_data, target_col, model_type)
            
        # 최종 모델 훈련
        final_model, metrics, predictions = self._train_final_model(
            train_data, val_data, test_data, target_col, model_type, best_params
        )
        
        # 결과 시각화
        if self.config.get('visualize_results', True):
            self._visualize_results(final_model, test_data, predictions, target_col)
            
        # 모델 저장
        if self.config.get('save_model', True):
            model_path = self._save_model(final_model, best_params)
            print(f"모델이 저장되었습니다: {model_path}")
            
        return TrainingResult(
            model=final_model,
            metrics=metrics,
            best_params=best_params,
            test_predictions=predictions.get('test_predictions'),
            validation_predictions=predictions.get('val_predictions'),
            training_history=final_model.history.history if hasattr(final_model, 'history') and final_model.history else None
        )
    
    def _preprocess_data(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """데이터 전처리"""
        # 데이터 복사
        df = data.copy()
        
        # 결측치 처리
        df = df.dropna()
        
        # 특성 선택
        if features is not None:
            target_col = self.config.get('target_column', 'close')
            if target_col not in features:
                features.append(target_col)
            df = df[features]
            
        return df
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """학습/검증/테스트 데이터 분할"""
        test_size = self.config.get('test_size', 0.2)
        val_size = self.config.get('validation_size', 0.2)
        
        # 시간 순서대로 분할
        n = len(data)
        test_end = n
        test_start = n - int(n * test_size)
        val_end = test_start
        val_start = val_end - int(n * val_size)
        
        train_data = data.iloc[:val_start].copy()
        val_data = data.iloc[val_start:val_end].copy()
        test_data = data.iloc[test_start:].copy()
        
        print(f"훈련 데이터: {len(train_data)} 샘플")
        print(f"검증 데이터: {len(val_data)} 샘플")
        print(f"테스트 데이터: {len(test_data)} 샘플")
        
        return train_data, val_data, test_data
    
    def _optimize_hyperparameters(self, train_data: pd.DataFrame, 
                                val_data: pd.DataFrame,
                                target_col: str,
                                model_type: ModelType) -> Dict[str, Any]:
        """하이퍼파라미터 최적화"""
        print("하이퍼파라미터 최적화 시작...")
        n_trials = self.config.get('optimization_trials', 30)
        sequence_length = self.config.get('sequence_length', 60)
        
        def objective(trial):
            # 최적화할 하이퍼파라미터 정의
            params = {
                'sequence_length': trial.suggest_int('sequence_length', 30, 120),
                'n_features': train_data.shape[1],
                'units1': trial.suggest_int('units1', 32, 128),
                'units2': trial.suggest_int('units2', 32, 128),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'batch_size': trial.suggest_int('batch_size', 16, 64),
                'epochs': 50,  # 최적화 중에는 에포크 수를 줄여 시간 단축
                'patience': 5
            }
            
            # 모델 생성
            predictor = TimeSeriesPredictorFactory.create_model(model_type, params)
            
            # 데이터 준비
            if train_data.shape[1] > 1:  # 다변량 시계열
                X_train, y_train = predictor.prepare_multivariate_data(
                    train_data, target_col=target_col
                )
                X_val, y_val = predictor.prepare_multivariate_data(
                    val_data, target_col=target_col
                )
            else:  # 단변량 시계열
                X_train, y_train = predictor.prepare_data(
                    train_data, target_col=target_col
                )
                X_val, y_val = predictor.prepare_data(
                    val_data, target_col=target_col
                )
            
            # 모델 훈련
            predictor.train(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                patience=params['patience']
            )
            
            # 검증 데이터로 평가
            metrics = predictor.evaluate(X_val, y_val)
            
            return metrics['mse']  # MSE 최소화
        
        # Optuna 스터디 생성 및 최적화 실행
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print("최적의 하이퍼파라미터:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
        # 최적 파라미터에 모델 유형별 파라미터 추가
        best_params = study.best_params.copy()
        best_params['n_features'] = train_data.shape[1]
        
        return best_params
    
    def _train_final_model(self, train_data: pd.DataFrame, 
                         val_data: pd.DataFrame,
                         test_data: pd.DataFrame,
                         target_col: str,
                         model_type: ModelType,
                         best_params: Optional[Dict[str, Any]] = None) -> Tuple[TimeSeriesPredictor, Dict[str, float], Dict[str, Any]]:
        """최종 모델 훈련"""
        print("최종 모델 훈련 시작...")
        
        # 설정값 및 기본값 설정
        sequence_length = self.config.get('sequence_length', 60)
        batch_size = self.config.get('batch_size', 32)
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 10)
        
        # 하이퍼파라미터 설정
        params = best_params.copy() if best_params else {}
        params.setdefault('sequence_length', sequence_length)
        params.setdefault('n_features', train_data.shape[1])
        params.setdefault('batch_size', batch_size)
        params.setdefault('epochs', epochs)
        params.setdefault('patience', patience)
        
        # 모델 생성
        predictor = TimeSeriesPredictorFactory.create_model(model_type, params)
        
        # 데이터 준비
        if train_data.shape[1] > 1:  # 다변량 시계열
            X_train, y_train = predictor.prepare_multivariate_data(
                train_data, target_col=target_col
            )
            X_val, y_val = predictor.prepare_multivariate_data(
                val_data, target_col=target_col
            )
            X_test, y_test = predictor.prepare_multivariate_data(
                test_data, target_col=target_col
            )
        else:  # 단변량 시계열
            X_train, y_train = predictor.prepare_data(
                train_data, target_col=target_col
            )
            X_val, y_val = predictor.prepare_data(
                val_data, target_col=target_col
            )
            X_test, y_test = predictor.prepare_data(
                test_data, target_col=target_col
            )
        
        # 모델 저장 경로 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.config.get('model_save_path', 'models/time_series/')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_type.name.lower()}_{timestamp}.h5")
        
        # 모델 훈련
        predictor.train(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            patience=params['patience'],
            model_path=model_path
        )
        
        # 검증 데이터로 평가
        val_metrics = predictor.evaluate(X_val, y_val)
        print("검증 데이터 성능:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.6f}")
            
        # 테스트 데이터로 평가
        test_metrics = predictor.evaluate(X_test, y_test)
        print("테스트 데이터 성능:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.6f}")
            
        # 예측
        val_pred = predictor.predict(X_val)
        test_pred = predictor.predict(X_test)
        
        predictions = {
            'val_predictions': val_pred,
            'test_predictions': test_pred,
            'y_val': predictor.scaler.inverse_transform(y_val),
            'y_test': predictor.scaler.inverse_transform(y_test)
        }
        
        return predictor, test_metrics, predictions
    
    def _visualize_results(self, model: TimeSeriesPredictor, 
                         test_data: pd.DataFrame, 
                         predictions: Dict[str, np.ndarray],
                         target_col: str) -> None:
        """결과 시각화"""
        # 훈련 과정 시각화
        model.plot_history()
        
        # 예측 결과 시각화
        y_test = predictions['y_test']
        test_pred = predictions['test_predictions']
        model.plot_predictions(y_test, test_pred, title='테스트 데이터 예측 결과')
        
        # 미래 예측 시각화
        if self.config.get('visualize_future', True):
            horizon = self.config.get('forecast_horizon', 30)
            
            if test_data.shape[1] > 1:  # 다변량 시계열
                last_sequence = test_data.iloc[-model.sequence_length:].values
            else:  # 단변량 시계열
                last_sequence = test_data[target_col].iloc[-model.sequence_length:].values.reshape(-1, 1)
                
            future_pred = model.forecast(last_sequence, horizon=horizon)
            
            plt.figure(figsize=(15, 6))
            plt.plot(range(len(y_test)), y_test, label='과거 데이터')
            plt.plot(range(len(y_test), len(y_test) + horizon), future_pred, label='미래 예측', color='red')
            plt.axvline(x=len(y_test), color='green', linestyle='--')
            plt.title('미래 예측 결과')
            plt.xlabel('시간')
            plt.ylabel('가격')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    def _save_model(self, model: TimeSeriesPredictor, best_params: Optional[Dict[str, Any]] = None) -> str:
        """모델 및 관련 정보 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.config.get('model_type', ModelType.LSTM).name.lower()
        model_dir = self.config.get('model_save_path', 'models/time_series/')
        
        # 디렉토리 생성
        os.makedirs(model_dir, exist_ok=True)
        
        # 모델 저장
        model_path = os.path.join(model_dir, f"{model_type}_{timestamp}.h5")
        scaler_path = os.path.join(model_dir, f"{model_type}_{timestamp}_scaler.pkl")
        model.save(model_path, scaler_path)
        
        # 하이퍼파라미터 및 설정 저장
        if best_params:
            params_path = os.path.join(model_dir, f"{model_type}_{timestamp}_params.json")
            with open(params_path, 'w') as f:
                json.dump(best_params, f, indent=4)
                
        # 설정 저장
        config_path = os.path.join(model_dir, f"{model_type}_{timestamp}_config.json")
        with open(config_path, 'w') as f:
            # config에서 ModelType 객체 처리
            config_copy = self.config.copy()
            if 'model_type' in config_copy:
                config_copy['model_type'] = config_copy['model_type'].name
            json.dump(config_copy, f, indent=4)
            
        return model_path
