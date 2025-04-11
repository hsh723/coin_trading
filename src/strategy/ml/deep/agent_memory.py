from collections import deque
import numpy as np
import torch

class AgentMemory:
    def __init__(self, capacity: int = 100000):
        self.memory = deque(maxlen=capacity)
        
    def add_experience(self, state, action, reward, next_state, done):
        """경험 추가"""
        self.memory.append((state, action, reward, next_state, done))
        
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """배치 샘플링"""
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.LongTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'next_states': torch.FloatTensor(next_states),
            'dones': torch.FloatTensor(dones)
        }
