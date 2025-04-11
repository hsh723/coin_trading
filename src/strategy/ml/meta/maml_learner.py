import torch
import torch.nn as nn
import higher

class MAMLLearner:
    def __init__(self, model: nn.Module, alpha: float = 0.01, beta: float = 0.001):
        self.model = model
        self.alpha = alpha  # 내부 학습률
        self.beta = beta   # 외부 학습률
        
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.beta
        )
