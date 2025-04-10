from typing import Dict, List
import hashlib
from dataclasses import dataclass

@dataclass
class IntegrityCheck:
    data_hash: str
    timestamp: float
    is_valid: bool
    error_messages: List[str]

class DataIntegrityChecker:
    def __init__(self, check_interval: int = 3600):
        self.check_interval = check_interval
        self.hash_history = {}
        
    async def check_integrity(self, data: bytes, metadata: Dict) -> IntegrityCheck:
        """데이터 무결성 검사"""
        current_hash = self._calculate_hash(data)
        stored_hash = self.hash_history.get(metadata['id'])
        
        is_valid = True
        errors = []
        
        if stored_hash and stored_hash != current_hash:
            is_valid = False
            errors.append(f"Data integrity violation detected for {metadata['id']}")
            
        self.hash_history[metadata['id']] = current_hash
        
        return IntegrityCheck(
            data_hash=current_hash,
            timestamp=pd.Timestamp.now().timestamp(),
            is_valid=is_valid,
            error_messages=errors
        )
