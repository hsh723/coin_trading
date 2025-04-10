from typing import Dict, List
import hashlib
from dataclasses import dataclass
import time

@dataclass
class Version:
    version_id: str
    timestamp: float
    checksum: str
    metadata: Dict
    parent_version: str = None

class DataVersionControl:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.versions = {}
        
    async def create_version(self, data: bytes, metadata: Dict) -> Version:
        """데이터 버전 생성"""
        version_id = self._generate_version_id()
        checksum = hashlib.sha256(data).hexdigest()
        
        version = Version(
            version_id=version_id,
            timestamp=time.time(),
            checksum=checksum,
            metadata=metadata,
            parent_version=self._get_latest_version()
        )
        
        await self._store_version(version, data)
        self.versions[version_id] = version
        return version
