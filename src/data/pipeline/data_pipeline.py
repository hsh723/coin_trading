from typing import Dict, List
import asyncio
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    collectors: List[str]
    processors: List[str]
    storage_type: str
    batch_size: int = 1000

class DataPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.collectors = self._initialize_collectors()
        self.processors = self._initialize_processors()
        self.storage = self._initialize_storage()
        
    async def run_pipeline(self, symbols: List[str]) -> bool:
        """데이터 파이프라인 실행"""
        try:
            raw_data = await self._collect_data(symbols)
            processed_data = await self._process_data(raw_data)
            await self._store_data(processed_data)
            return True
        except Exception as e:
            self._handle_pipeline_error(e)
            return False
