from typing import Dict, Any
import zlib
import json
from dataclasses import dataclass

@dataclass
class CompressionStats:
    original_size: int
    compressed_size: int
    compression_ratio: float
    method: str

class DataCompressor:
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        
    async def compress_data(self, data: Dict[str, Any]) -> bytes:
        """데이터 압축"""
        json_data = json.dumps(data)
        compressed = zlib.compress(json_data.encode(), self.compression_level)
        
        self.stats = CompressionStats(
            original_size=len(json_data),
            compressed_size=len(compressed),
            compression_ratio=len(compressed) / len(json_data),
            method='zlib'
        )
        
        return compressed
