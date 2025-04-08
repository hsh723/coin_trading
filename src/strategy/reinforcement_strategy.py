import numpy as np
from typing import Dict, List
import torch
import torch.nn as nn
from stable_baselines3 import PPO

class ReinforcementStrategy:
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
