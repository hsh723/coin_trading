from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, config: Dict = None):
        self.config = config or {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'memory_size': 10000,
            'batch_size': 64
        }
        
        self.policy_net = self._build_network(state_size, action_size)
        self.target_net = self._build_network(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    async def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(torch.FloatTensor(state)).max(1)[1].item()
        return np.random.randint(self.action_size)
