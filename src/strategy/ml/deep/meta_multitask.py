import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class MetaMultiTaskLearner(nn.Module):
    def __init__(self, input_dim: int, task_dims: Dict[str, int]):
        super().__init__()
        
        # 공유 인코더
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 태스크별 어댑터
        self.task_adapters = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, dim)
            ) for task, dim in task_dims.items()
        })
        
        # 메타 컨트롤러
        self.meta_controller = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
