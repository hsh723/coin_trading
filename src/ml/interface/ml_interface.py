#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
머신러닝 시스템 통합 인터페이스
시계열 예측, 시장 상태 분류, 앙상블 모델, 온라인 학습 기능 등을 쉽게 활용할 수 있는 인터페이스 제공
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import json

from ..forecasting.time_series_predictor import TimeSeriesPredictor
from ..forecasting.model_factory import TimeSeriesPredictorFactory, ModelType
from ..ensemble.time_series_ensemble import TimeSeriesEnsemble
from ..online.time_series_online_learner import OnlineLearner
from ..classification.market_state_classifier import MarketStateClassifier
from ..training_pipeline import ModelTrainingPipeline

class MLInterface:
    """머신러닝 시스템 통합 인터페이스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로 (None이면 기본 설정 사용)
        """
        self.logger = logging.getLogger(__name__)
        
        # 기본 설정
        self.config = {
            'model_save_path': 'models/',
            'results_path': 'results/',
            'default_features': [
                'close', 'volume', 'ma7', 'ma14', 'ma30', 
                'rsi', 'macd', 'macd_hist', 'bb_width'
            ],
            'time_series': {
                'sequence_length': 48,
                'model_type': 'ensemble',  # 'single' 또는 'ensemble'
                'prediction_horizon': 24
            },
            'classification': {
                'class_thresholds': {
                    'bullish': 0.03,    # 3% 이상 상승
                    'bearish': -0.03,   # 3% 이상 하락
                    'sideways': 0.01    # -1% ~ 1% 횡보
                },
                'prediction_period': 24,  # 24시간 후 예측
                'model_type': 'random_forest'
            },
            'online_learning': {
                'update_frequency': 24,  # 시간 단위
                'min_data_points': 48,   # 최소 데이터 포인트
                'max_data_points': 1000, # 최대 데이터 포인트
                'fine_tune_epochs': 10,
                'full_retrain_threshold': 0.05
            }
        }
        
        # 설정 파일 로드
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # 설정 병합
                self._merge_configs(self.config, user_config)
        
        # 디렉터리 생성
        os.makedirs(self.config['model_save_path'], exist_ok=True)
        os.makedirs(self.config['results_path'], exist_ok=True)
        
        # 모델 인스턴스
        self.time_series_model = None
        self.classification_model = None
        self.online_learner = None
    
    def _merge_configs(self, base_config: Dict, user_config: Dict) -> None:
        """
        설정 병합
        
        Args:
            base_config: 기본 설정
            user_config: 사용자 설정
        """
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value
    
    def prepare_data(self, data: pd.DataFrame, add_features: bool = True) -> pd.DataFrame:
        """
        데이터 전처리 및 기술적 지표 추가
        
        Args:
            data: 원시 가격 데이터 (OHLCV)
            add_features: 기술적 지표 추가 여부
            
        Returns:
            전처리된 데이터
        """
        # 데이터 형식 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"데이터에 필수 열이 누락되었습니다: {missing_columns}")
            raise ValueError(f"데이터에 필수 열이 누락되었습니다: {missing_columns}")
        
        df = data.copy()
        
        # 기술적 지표 추가
        if add_features:
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
            
            # 가격 모멘텀
            df['momentum'] = df['close'].pct_change(periods=10)
            
            # 누락된 값 제거
            df.dropna(inplace=True)
        
        return df
    
    def train_time_series_model(self, data: pd.DataFrame, 
                              features: List[str] = None, 
                              target_col: str = 'close', 
                              model_config: Dict = None) -> Dict[str, Any]:
        """
        시계열 예측 모델 훈련
        
        Args:
            data: 훈련 데이터
            features: 특성 목록
            target_col: 타겟 열
            model_config: 모델 설정
            
        Returns:
            훈련 결과
        """
        # 특성 목록 설정
        if features is None:
            features = self.config['default_features']
        
        # 필수 특성 확인
        if target_col not in data.columns:
            self.logger.error(f"타겟 열이 데이터에 없습니다: {target_col}")
            raise ValueError(f"타겟 열이 데이터에 없습니다: {target_col}")
        
        # 모델 설정
        ts_config = self.config['time_series'].copy()
        if model_config:
            ts_config.update(model_config)
        
        # 훈련 파이프라인 설정
        pipeline_config = {
            'sequence_length': ts_config.get('sequence_length', 48),
            'target_column': target_col,
            'model_type': ModelType.LSTM if ts_config.get('model_type') == 'single' else ModelType.LSTM,
            'optimize_hyperparameters': ts_config.get('optimize_hyperparameters', True),
            'optimization_trials': ts_config.get('optimization_trials', 20),
            'batch_size': ts_config.get('batch_size', 32),
            'epochs': ts_config.get('epochs', 100),
            'patience': ts_config.get('patience', 10),
            'validation_size': ts_config.get('validation_size', 0.2),
            'test_size': ts_config.get('test_size', 0.2),
            'save_model': True,
            'visualize_results': True,
            'model_save_path': os.path.join(self.config['model_save_path'], 'time_series/'),
            'forecast_horizon': ts_config.get('prediction_horizon', 24)
        }
        
        # 앙상블 모델 설정
        if ts_config.get('model_type') == 'ensemble':
            # 앙상블 모델 설정
            ensemble_config = {
                'sequence_length': ts_config.get('sequence_length', 48),
                'n_features': len(features),
                'ensemble_method': ts_config.get('ensemble_method', 'optimal'),
                'model_types': [ModelType.LSTM, ModelType.GRU, ModelType.BIDIRECTIONAL_LSTM]
            }
            
            # 훈련 파이프라인 수정
            pipeline_config['is_ensemble'] = True
            pipeline_config['ensemble_config'] = ensemble_config
        
        # 훈련 파이프라인 생성 및 실행
        pipeline = ModelTrainingPipeline(pipeline_config)
        result = pipeline.run_pipeline(data, features=features, target=target_col)
        
        # 모델 저장
        self.time_series_model = result.model
        
        # 결과 반환
        return {
            'metrics': result.metrics,
            'best_params': result.best_params,
            'model': self.time_series_model
        }
    
    def predict_time_series(self, data: pd.DataFrame, 
                          features: List[str] = None, 
                          target_col: str = 'close',
                          horizon: int = None) -> Dict[str, Any]:
        """
        시계열 예측 수행
        
        Args:
            data: 입력 데이터
            features: 특성 목록
            target_col: 타겟 열
            horizon: 예측 기간
            
        Returns:
            예측 결과
        """
        if self.time_series_model is None:
            self.logger.error("시계열 모델이 훈련되지 않았습니다.")
            raise ValueError("시계열 모델이 훈련되지 않았습니다.")
        
        # 특성 목록 설정
        if features is None:
            features = self.config['default_features']
        
        # 예측 기간 설정
        if horizon is None:
            horizon = self.config['time_series'].get('prediction_horizon', 24)
        
        # 데이터 준비
        sequence_length = self.config['time_series'].get('sequence_length', 48)
        
        if len(data) < sequence_length:
            self.logger.error(f"데이터가 부족합니다. 최소 {sequence_length}개 필요, 현재 {len(data)}개.")
            raise ValueError(f"데이터가 부족합니다. 최소 {sequence_length}개 필요, 현재 {len(data)}개.")
        
        # 앙상블 모델인지 단일 모델인지 확인
        is_ensemble = isinstance(self.time_series_model, TimeSeriesEnsemble)
        
        try:
            # 마지막 시퀀스 준비
            last_sequence = data[features].iloc[-sequence_length:].values
            
            # 예측 수행
            if is_ensemble:
                predictions = self.time_series_model.forecast(last_sequence, horizon)
            else:
                predictions = self.time_series_model.forecast(last_sequence, horizon)
            
            # 결과 준비
            forecast_times = [data.index[-1] + timedelta(hours=i+1) for i in range(horizon)]
            forecast_values = [float(pred[0]) for pred in predictions]
            
            result = {
                'forecast_times': forecast_times,
                'forecast_values': forecast_values,
                'horizon': horizon
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {e}")
            raise
    
    def train_market_state_classifier(self, data: pd.DataFrame, 
                                    features: List[str] = None,
                                    model_config: Dict = None) -> Dict[str, Any]:
        """
        시장 상태 분류 모델 훈련
        
        Args:
            data: 훈련 데이터
            features: 특성 목록
            model_config: 모델 설정
            
        Returns:
            훈련 결과
        """
        # 특성 목록 설정
        if features is None:
            features = self.config['default_features']
        
        # 모델 설정
        cls_config = self.config['classification'].copy()
        if model_config:
            cls_config.update(model_config)
        
        # 분류기 생성
        save_path = os.path.join(self.config['model_save_path'], 'market_state/')
        os.makedirs(save_path, exist_ok=True)
        
        classifier_config = {
            'class_thresholds': cls_config.get('class_thresholds', {
                'bullish': 0.03,
                'bearish': -0.03,
                'sideways': 0.01
            }),
            'prediction_period': cls_config.get('prediction_period', 24),
            'model_type': cls_config.get('model_type', 'random_forest'),
            'feature_window': cls_config.get('feature_window', 14),
            'model_save_path': save_path
        }
        
        # 분류기 초기화
        self.classification_model = MarketStateClassifier(classifier_config)
        
        # 데이터 준비 및 훈련
        X, y = self.classification_model.prepare_data(data, features=features)
        metrics = self.classification_model.train(X, y)
        
        # 모델 저장
        model_path = self.classification_model.save()
        
        # 결과 반환
        return {
            'metrics': metrics,
            'model': self.classification_model,
            'model_path': model_path
        }
    
    def predict_market_state(self, data: pd.DataFrame, 
                           features: List[str] = None) -> Dict[str, Any]:
        """
        시장 상태 예측
        
        Args:
            data: 입력 데이터
            features: 특성 목록
            
        Returns:
            예측 결과
        """
        if self.classification_model is None:
            self.logger.error("시장 상태 모델이 훈련되지 않았습니다.")
            raise ValueError("시장 상태 모델이 훈련되지 않았습니다.")
        
        # 특성 목록 설정
        if features is None:
            features = self.config['default_features']
        
        try:
            # 예측 수행
            predictions = self.classification_model.predict(data[features])
            
            # 결과 준비
            last_pred = predictions.iloc[-1]
            result = {
                'predicted_state': last_pred['predicted_state'],
                'probabilities': {
                    'bullish': last_pred.get('prob_bullish', 0),
                    'bearish': last_pred.get('prob_bearish', 0),
                    'sideways': last_pred.get('prob_sideways', 0),
                    'neutral': last_pred.get('prob_neutral', 0)
                },
                'timestamp': data.index[-1]
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {e}")
            raise
    
    def setup_online_learning(self, data: pd.DataFrame, 
                            features: List[str] = None,
                            target_col: str = 'close',
                            model_path: str = None,
                            online_config: Dict = None) -> OnlineLearner:
        """
        온라인 학습 시스템 설정
        
        Args:
            data: 초기 학습 데이터
            features: 특성 목록
            target_col: 타겟 열
            model_path: 기존 모델 경로
            online_config: 온라인 학습 설정
            
        Returns:
            온라인 학습기
        """
        # 특성 목록 설정
        if features is None:
            features = self.config['default_features']
        
        # 모델 설정
        ol_config = self.config['online_learning'].copy()
        if online_config:
            ol_config.update(online_config)
        
        # 온라인 학습기 설정
        learner_config = {
            'model_type': 'ensemble',
            'model_path': model_path,
            'update_frequency': ol_config.get('update_frequency', 24),
            'min_data_points': ol_config.get('min_data_points', 48),
            'max_data_points': ol_config.get('max_data_points', 1000),
            'sequence_length': self.config['time_series'].get('sequence_length', 48),
            'n_features': len(features),
            'target_column': target_col,
            'model_save_path': os.path.join(self.config['model_save_path'], 'online/'),
            'fine_tune_epochs': ol_config.get('fine_tune_epochs', 10),
            'full_retrain_threshold': ol_config.get('full_retrain_threshold', 0.05),
            'ensemble_config': {
                'sequence_length': self.config['time_series'].get('sequence_length', 48),
                'n_features': len(features),
                'ensemble_method': 'optimal',
                'model_types': [ModelType.LSTM, ModelType.GRU, ModelType.BIDIRECTIONAL_LSTM]
            }
        }
        
        # 온라인 학습기 초기화
        self.online_learner = OnlineLearner(learner_config)
        
        # 초기 학습 (이미 모델이 있으면 건너뜀)
        if model_path is None or not os.path.exists(model_path):
            self.logger.info("온라인 학습기 초기 학습 시작...")
            update_result = self.online_learner.update(data, features)
            
            if update_result:
                self.logger.info(f"초기 학습 완료 - RMSE: {update_result.metrics_after['rmse']:.6f}")
            else:
                self.logger.warning("초기 학습이 수행되지 않았습니다.")
        
        return self.online_learner
    
    def update_online_model(self, data: pd.DataFrame, 
                           features: List[str] = None) -> Dict[str, Any]:
        """
        온라인 모델 업데이트
        
        Args:
            data: 새 데이터
            features: 특성 목록
            
        Returns:
            업데이트 결과
        """
        if self.online_learner is None:
            self.logger.error("온라인 학습기가 초기화되지 않았습니다.")
            raise ValueError("온라인 학습기가 초기화되지 않았습니다.")
        
        # 특성 목록 설정
        if features is None:
            features = self.config['default_features']
        
        # 업데이트 수행
        update_result = self.online_learner.update(data, features)
        
        if update_result:
            result = {
                'update_time': update_result.update_time,
                'metrics_before': update_result.metrics_before,
                'metrics_after': update_result.metrics_after,
                'performance_change': update_result.performance_change,
                'data_points': update_result.data_points_used
            }
            return result
        else:
            return {
                'update_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'skipped',
                'reason': '업데이트 조건이 충족되지 않았습니다.'
            }
    
    def online_forecast(self, data: pd.DataFrame, 
                       features: List[str] = None,
                       horizon: int = None) -> Dict[str, Any]:
        """
        온라인 모델을 사용한 예측
        
        Args:
            data: 입력 데이터
            features: 특성 목록
            horizon: 예측 기간
            
        Returns:
            예측 결과
        """
        if self.online_learner is None:
            self.logger.error("온라인 학습기가 초기화되지 않았습니다.")
            raise ValueError("온라인 학습기가 초기화되지 않았습니다.")
        
        # 특성 목록 설정
        if features is None:
            features = self.config['default_features']
        
        # 예측 기간 설정
        if horizon is None:
            horizon = self.config['time_series'].get('prediction_horizon', 24)
        
        # 예측 수행
        try:
            forecasts = self.online_learner.forecast(data, horizon=horizon, features=features)
            
            # 결과 준비
            forecast_times = [data.index[-1] + timedelta(hours=i+1) for i in range(horizon)]
            forecast_values = [float(pred[0]) for pred in forecasts]
            
            result = {
                'forecast_times': forecast_times,
                'forecast_values': forecast_values,
                'horizon': horizon
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {e}")
            raise
    
    def load_time_series_model(self, model_path: str, scaler_path: str = None) -> None:
        """
        시계열 모델 로드
        
        Args:
            model_path: 모델 파일 경로
            scaler_path: 스케일러 파일 경로
        """
        try:
            # 앙상블 모델인지 단일 모델인지 확인
            if 'ensemble' in model_path:
                # TODO: 앙상블 모델 로드 구현
                self.logger.warning("앙상블 모델 로드는 아직 구현되지 않았습니다.")
            else:
                # 단일 모델 로드
                if scaler_path is None:
                    scaler_path = model_path.replace('.h5', '_scaler.pkl')
                
                sequence_length = self.config['time_series'].get('sequence_length', 48)
                n_features = len(self.config['default_features'])
                
                self.time_series_model = TimeSeriesPredictor.load(
                    model_path, scaler_path, 
                    sequence_length=sequence_length, 
                    n_features=n_features
                )
                
                self.logger.info(f"시계열 모델 로드 완료: {model_path}")
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
    
    def load_market_state_model(self, model_path: str) -> None:
        """
        시장 상태 모델 로드
        
        Args:
            model_path: 모델 파일 경로
        """
        try:
            cls_config = self.config['classification'].copy()
            
            classifier_config = {
                'class_thresholds': cls_config.get('class_thresholds', {
                    'bullish': 0.03,
                    'bearish': -0.03,
                    'sideways': 0.01
                }),
                'prediction_period': cls_config.get('prediction_period', 24),
                'model_type': cls_config.get('model_type', 'random_forest'),
                'feature_window': cls_config.get('feature_window', 14)
            }
            
            self.classification_model = MarketStateClassifier(classifier_config)
            self.classification_model.load_model(model_path)
            
            self.logger.info(f"시장 상태 모델 로드 완료: {model_path}")
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
    
    def visualize_predictions(self, data: pd.DataFrame, predictions: Dict, 
                            title: str = '가격 예측', 
                            save_path: Optional[str] = None) -> None:
        """
        예측 결과 시각화
        
        Args:
            data: 과거 데이터
            predictions: 예측 결과
            title: 그래프 제목
            save_path: 저장 경로
        """
        plt.figure(figsize=(15, 6))
        
        # 과거 데이터 플롯
        plt.plot(data.index, data['close'], label='실제 가격', color='blue')
        
        # 예측 데이터 플롯
        forecast_times = predictions['forecast_times']
        forecast_values = predictions['forecast_values']
        
        plt.plot(forecast_times, forecast_values, label='예측 가격', color='red', linestyle='--')
        
        # 분리선
        plt.axvline(x=data.index[-1], color='green', linestyle='-.')
        
        plt.title(title)
        plt.xlabel('시간')
        plt.ylabel('가격')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 저장
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"그래프 저장됨: {save_path}")
        
        plt.show()
    
    def visualize_market_state(self, data: pd.DataFrame, predictions: pd.DataFrame,
                             save_path: Optional[str] = None) -> None:
        """
        시장 상태 시각화
        
        Args:
            data: 가격 데이터
            predictions: 예측 결과
            save_path: 저장 경로
        """
        plt.figure(figsize=(15, 8))
        
        # 가격 차트
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(data.index, data['close'], label='가격')
        ax1.set_title('가격 및 시장 상태')
        ax1.set_ylabel('가격')
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
        
        ax2.set_xlabel('시간')
        ax2.set_ylabel('시장 상태')
        ax2.set_yticks([])
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 저장
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"그래프 저장됨: {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, file_name: str) -> str:
        """
        결과 저장
        
        Args:
            results: 저장할 결과
            file_name: 파일 이름
            
        Returns:
            저장 경로
        """
        os.makedirs(self.config['results_path'], exist_ok=True)
        
        # 결과 파일 경로
        result_path = os.path.join(self.config['results_path'], file_name)
        
        # JSON으로 저장
        with open(result_path, 'w') as f:
            # datetime 객체 처리
            def json_serial(obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.strftime('%Y-%m-%d %H:%M:%S')
                raise TypeError(f"Type {type(obj)} not serializable")
            
            json.dump(results, f, default=json_serial, indent=4)
        
        self.logger.info(f"결과 저장됨: {result_path}")
        
        return result_path 