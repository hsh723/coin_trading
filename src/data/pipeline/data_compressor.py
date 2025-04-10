from typing import Dict, Union
import lz4.frame
import json
from dataclasses import dataclass

@dataclass
class CompressionResult:
    original_size: int
    compressed_size: int
    compression_ratio: float
    format: str

class DataCompressor:
    def __init__(self, compression_level: int = 3):
        self.compression_level = compression_level
        
    async def compress_data(self, data: Union[bytes, str, Dict]) -> CompressionResult:
        """데이터 압축"""
        if isinstance(data, (str, Dict)):
            data = json.dumps(data).encode()
            
        compressed = lz4.frame.compress(
            data,
            compression_level=self.compression_level
        )
        
        return CompressionResult(
            original_size=len(data),
            compressed_size=len(compressed),
            compression_ratio=len(compressed) / len(data),
            format='lz4'
        )
