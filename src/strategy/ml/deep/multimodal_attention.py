import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class MultiModalAttention(nn.Module):
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {
            'price_dim': 64,
            'volume_dim': 32,
            'sentiment_dim': 32,
            'num_heads': 8
        }
        
        # 모달리티별 인코더
        self.price_encoder = nn.Linear(1, self.config['price_dim'])
        self.volume_encoder = nn.Linear(1, self.config['volume_dim'])
        self.sentiment_encoder = nn.Linear(1, self.config['sentiment_dim'])
        
        # 크로스 어텐션 레이어
        total_dim = self.config['price_dim'] + self.config['volume_dim'] + self.config['sentiment_dim']
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=self.config['num_heads']
        )
