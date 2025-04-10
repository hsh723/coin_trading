from typing import Dict, Optional
from dataclasses import dataclass
import hmac
import hashlib

@dataclass
class SecurityContext:
    api_key: str
    signature: str
    timestamp: float
    nonce: str
    permissions: List[str]

class ApiSecurityManager:
    def __init__(self, security_config: Dict = None):
        self.config = security_config or {
            'signature_method': 'hmac-sha256',
            'nonce_window': 60
        }
        
    async def secure_request(self, request_data: Dict, 
                           api_credentials: Dict) -> SecurityContext:
        """API 요청 보안 처리"""
        timestamp = time.time()
        nonce = self._generate_nonce()
        
        return SecurityContext(
            api_key=api_credentials['api_key'],
            signature=self._generate_signature(request_data, api_credentials['secret']),
            timestamp=timestamp,
            nonce=nonce,
            permissions=api_credentials.get('permissions', [])
        )
