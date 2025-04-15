import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Type
from enum import Enum, auto
from .time_series_predictor import TimeSeriesPredictor
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Conv1D, GlobalMaxPooling1D, Bidirectional

class ModelType(Enum):
    LSTM = auto()
    GRU = auto()
    SIMPLE_RNN = auto()
    CNN_LSTM = auto()
    BIDIRECTIONAL_LSTM = auto()

class TimeSeriesPredictorFactory:
    """다양한 시계열 예측 모델을 생성하는 팩토리 클래스"""
    
    @staticmethod
    def create_model(model_type: ModelType, config: Dict[str, Any] = None) -> TimeSeriesPredictor:
        """지정된 유형의 시계열 예측 모델 생성
        
        Args:
            model_type: 모델 유형 (ModelType Enum)
            config: 모델 구성 파라미터
            
        Returns:
            TimeSeriesPredictor: 구성된 예측 모델
        """
        if config is None:
            config = {}
            
        sequence_length = config.get('sequence_length', 60)
        n_features = config.get('n_features', 1)
        
        if model_type == ModelType.LSTM:
            return TimeSeriesPredictorFactory._create_lstm_model(sequence_length, n_features, config)
        elif model_type == ModelType.GRU:
            return TimeSeriesPredictorFactory._create_gru_model(sequence_length, n_features, config)
        elif model_type == ModelType.SIMPLE_RNN:
            return TimeSeriesPredictorFactory._create_simple_rnn_model(sequence_length, n_features, config)
        elif model_type == ModelType.CNN_LSTM:
            return TimeSeriesPredictorFactory._create_cnn_lstm_model(sequence_length, n_features, config)
        elif model_type == ModelType.BIDIRECTIONAL_LSTM:
            return TimeSeriesPredictorFactory._create_bidirectional_lstm_model(sequence_length, n_features, config)
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {model_type}")
    
    @staticmethod
    def _create_lstm_model(sequence_length: int, n_features: int, config: Dict[str, Any]) -> TimeSeriesPredictor:
        """기본 LSTM 모델 생성"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        units1 = config.get('units1', 50)
        units2 = config.get('units2', 50)
        dropout_rate = config.get('dropout_rate', 0.2)
        
        predictor = TimeSeriesPredictor(sequence_length=sequence_length, n_features=n_features)
        predictor.model = Sequential([
            LSTM(units1, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(dropout_rate),
            LSTM(units2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25),
            Dense(1)
        ])
        predictor.model.compile(optimizer='adam', loss='mse')
        
        return predictor
    
    @staticmethod
    def _create_gru_model(sequence_length: int, n_features: int, config: Dict[str, Any]) -> TimeSeriesPredictor:
        """GRU 기반 모델 생성"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout
        
        units1 = config.get('units1', 50)
        units2 = config.get('units2', 50)
        dropout_rate = config.get('dropout_rate', 0.2)
        
        predictor = TimeSeriesPredictor(sequence_length=sequence_length, n_features=n_features)
        predictor.model = Sequential([
            GRU(units1, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(dropout_rate),
            GRU(units2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25),
            Dense(1)
        ])
        predictor.model.compile(optimizer='adam', loss='mse')
        
        return predictor
    
    @staticmethod
    def _create_simple_rnn_model(sequence_length: int, n_features: int, config: Dict[str, Any]) -> TimeSeriesPredictor:
        """SimpleRNN 기반 모델 생성"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
        
        units1 = config.get('units1', 50)
        units2 = config.get('units2', 50)
        dropout_rate = config.get('dropout_rate', 0.2)
        
        predictor = TimeSeriesPredictor(sequence_length=sequence_length, n_features=n_features)
        predictor.model = Sequential([
            SimpleRNN(units1, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(dropout_rate),
            SimpleRNN(units2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25),
            Dense(1)
        ])
        predictor.model.compile(optimizer='adam', loss='mse')
        
        return predictor
    
    @staticmethod
    def _create_cnn_lstm_model(sequence_length: int, n_features: int, config: Dict[str, Any]) -> TimeSeriesPredictor:
        """CNN + LSTM 하이브리드 모델 생성"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
        
        filters = config.get('filters', 64)
        kernel_size = config.get('kernel_size', 3)
        lstm_units = config.get('lstm_units', 50)
        dropout_rate = config.get('dropout_rate', 0.2)
        
        predictor = TimeSeriesPredictor(sequence_length=sequence_length, n_features=n_features)
        predictor.model = Sequential([
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', 
                   input_shape=(sequence_length, n_features)),
            MaxPooling1D(pool_size=2),
            Dropout(dropout_rate),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25),
            Dense(1)
        ])
        predictor.model.compile(optimizer='adam', loss='mse')
        
        return predictor
    
    @staticmethod
    def _create_bidirectional_lstm_model(sequence_length: int, n_features: int, config: Dict[str, Any]) -> TimeSeriesPredictor:
        """양방향 LSTM 모델 생성"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
        
        units = config.get('units', 50)
        dropout_rate = config.get('dropout_rate', 0.2)
        
        predictor = TimeSeriesPredictor(sequence_length=sequence_length, n_features=n_features)
        predictor.model = Sequential([
            Bidirectional(LSTM(units, return_sequences=True), 
                          input_shape=(sequence_length, n_features)),
            Dropout(dropout_rate),
            Bidirectional(LSTM(units, return_sequences=False)),
            Dropout(dropout_rate),
            Dense(25),
            Dense(1)
        ])
        predictor.model.compile(optimizer='adam', loss='mse')
        
        return predictor 