from typing import Dict, Optional
import hashlib
from dataclasses import dataclass

@dataclass
class Version:
    version_id: str
    timestamp: float
    checksum: str
    metadata: Dict

class DataVersioner:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.versions = {}
        
    async def create_version(self, data: bytes, metadata: Dict) -> Version:
        """데이터 버전 생성"""
        version_id = self._generate_version_id()
        checksum = self._calculate_checksum(data)
        
        version = Version(
            version_id=version_id,
            timestamp=time.time(),
            checksum=checksum,
            metadata=metadata
        )
        
        await self._store_version(version, data)
        return version
