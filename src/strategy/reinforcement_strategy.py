import numpy as np
from typing import Dict, List
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from dataclasses import dataclass

@dataclass
class RLState:
    features: np.ndarray
    position: int
    unrealized_pnl: float
    market_state: Dict[str, float]

class ReinforcementStrategy:
    def __init__(self, model_config: Dict = None):
        self.config = model_config or {
            'state_size': 10,
            'action_size': 3,  # buy, sell, hold
            'learning_rate': 0.001,
            'gamma': 0.99
        }
        self.model = self._build_model()
        
    async def get_action(self, state: RLState) -> Dict:
        """강화학습 기반 행동 결정"""
        state_vector = self._preprocess_state(state)
        q_values = self.model.predict(state_vector)
        
        action = np.argmax(q_values)
        return {
            'action': ['buy', 'sell', 'hold'][action],
            'confidence': float(q_values[action]),
            'q_values': q_values.tolist()
        }

    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.model = PPO(
            'MlpPolicy',
            env=self._create_trading_env(),
            learning_rate=config.get('learning_rate', 3e-4),
            n_steps=config.get('n_steps', 2048),
            batch_size=config.get('batch_size', 64),
            verbose=1
        )
        
    def train(self, market_data: pd.DataFrame, n_epochs: int = 10):
        """모델 학습"""
        for epoch in range(n_epochs):
            self.model.learn(
                total_timesteps=10000,
                callback=self._create_callback()
            )
