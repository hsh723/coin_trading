import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class HybridAttentionRL(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
        super().__init__()
        
        self.config = config or {
            'hidden_dim': 256,
            'num_heads': 8,
            'dropout': 0.1,
            'num_layers': 3
        }
        
        # 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, self.config['hidden_dim']),
            nn.LayerNorm(self.config['hidden_dim']),
            nn.GELU()
        )
        
        # 멀티헤드 어텐션
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.config['hidden_dim'],
                num_heads=self.config['num_heads'],
                dropout=self.config['dropout']
            ) for _ in range(self.config['num_layers'])
        ])
        
        # 정책 및 가치 헤드
        self.policy_head = nn.Sequential(
            nn.Linear(self.config['hidden_dim'], action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.config['hidden_dim'], 1)
        )
