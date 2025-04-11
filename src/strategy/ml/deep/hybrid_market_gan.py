import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridMarketGAN:
    def __init__(self, seq_length: int, latent_dim: int = 100):
        super().__init__()
        self.generator = Generator(latent_dim, seq_length)
        self.discriminator = Discriminator(seq_length)
        self.attention_bridge = AttentionBridge(128)
        
    async def generate_samples(self, 
                             market_conditions: torch.Tensor,
                             num_samples: int) -> torch.Tensor:
        """시장 조건에 기반한 샘플 생성"""
        noise = torch.randn(num_samples, self.latent_dim)
        conditions = self.attention_bridge(market_conditions)
        return self.generator(torch.cat([noise, conditions], dim=1))
