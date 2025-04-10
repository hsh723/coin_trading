from typing import Dict, Optional
from dataclasses import dataclass
import hmac
import hashlib
import time

@dataclass
class AuthenticationInfo:
    exchange_id: str
    auth_type: str
    signature: str
    timestamp: float
    expiry: Optional[float]

class ApiAuthenticationManager:
    def __init__(self, auth_config: Dict = None):
        self.config = auth_config or {
            'signature_validity': 30,  # seconds
            'hmac_algorithm': 'sha256'
        }
        
    async def generate_authentication(self, 
                                   exchange_id: str, 
                                   credentials: Dict) -> AuthenticationInfo:
        """API 인증 정보 생성"""
        timestamp = time.time()
        message = self._create_signature_message(exchange_id, timestamp)
        signature = self._sign_message(message, credentials['secret'])
        
        return AuthenticationInfo(
            exchange_id=exchange_id,
            auth_type='hmac',
            signature=signature,
            timestamp=timestamp,
            expiry=timestamp + self.config['signature_validity']
        )
