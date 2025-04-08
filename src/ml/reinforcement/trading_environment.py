import gym
import numpy as np
from typing import Tuple, Dict
from gym import spaces

class TradingEnvironment(gym.Env):
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0):
        super().__init__()
        self.data = data
        self.initial_balance = initial_balance
        
        # 행동 공간: [매수 비율, 매도 비율]
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([1, 1]), 
            dtype=np.float32
        )
        
        # 관찰 공간: [가격 데이터, 기술적 지표, 포지션 정보]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(20,), 
            dtype=np.float32
        )
