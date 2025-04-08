import numpy as np
import pandas as pd
from typing import Dict, List
import tensorflow as tf
from .base import BaseStrategy

class AIStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.model = self._build_model(config['model_params'])
        self.lookback = config.get('lookback_period', 60)
        self.feature_columns = config.get('feature_columns', ['close', 'volume'])
        
    def _build_model(self, params: Dict) -> tf.keras.Model:
        """LSTM 모델 구축"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=params.get('lstm_units', 50),
                               input_shape=(self.lookback, len(self.feature_columns))),
            tf.keras.layers.Dense(units=params.get('dense_units', 32),
                                activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
