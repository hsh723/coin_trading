from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class ExportFormat:
    format_type: str
    schema: Dict
    compression: bool

class MetricsExporter:
    def __init__(self, export_config: Dict = None):
        self.config = export_config or {
            'formats': ['csv', 'json'],
            'compression': True,
            'batch_size': 1000
        }
        
    async def export_metrics(self, metrics: List[Dict], 
                           format_type: str) -> str:
        """메트릭스 데이터 내보내기"""
        if format_type not in self.config['formats']:
            raise ValueError(f"Unsupported format: {format_type}")
            
        export_path = self._generate_export_path(format_type)
        
        if format_type == 'csv':
            await self._export_to_csv(metrics, export_path)
        elif format_type == 'json':
            await self._export_to_json(metrics, export_path)
            
        if self.config['compression']:
            export_path = await self._compress_export(export_path)
            
        return export_path
