from typing import Dict, Optional
from dataclasses import dataclass
import json
from cryptography.fernet import Fernet

@dataclass
class ApiKeyInfo:
    exchange_id: str
    key_status: str
    permissions: List[str]
    last_used: float
    rate_limit_info: Dict

class ApiKeyManager:
    def __init__(self, encryption_key: bytes = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.api_keys = {}
        
    async def add_api_key(self, exchange_id: str, 
                         credentials: Dict) -> ApiKeyInfo:
        """API 키 추가 및 암호화"""
        encrypted_key = self.fernet.encrypt(
            json.dumps(credentials).encode()
        )
        
        key_info = ApiKeyInfo(
            exchange_id=exchange_id,
            key_status='active',
            permissions=credentials.get('permissions', []),
            last_used=time.time(),
            rate_limit_info={}
        )
        
        self.api_keys[exchange_id] = {
            'encrypted': encrypted_key,
            'info': key_info
        }
        
        return key_info
