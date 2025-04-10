from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class PipelineReport:
    pipeline_id: str
    execution_summary: Dict
    performance_metrics: Dict
    error_summary: List[Dict]
    recommendations: List[str]

class PipelineReporter:
    def __init__(self, report_config: Dict = None):
        self.config = report_config or {
            'include_metrics': True,
            'include_errors': True,
            'include_recommendations': True
        }
        
    async def generate_report(self, pipeline_id: str) -> PipelineReport:
        """파이프라인 리포트 생성"""
        execution_data = await self._collect_execution_data(pipeline_id)
        metrics = await self._calculate_metrics(execution_data)
        errors = await self._analyze_errors(execution_data)
        
        return PipelineReport(
            pipeline_id=pipeline_id,
            execution_summary=self._create_summary(execution_data),
            performance_metrics=metrics,
            error_summary=errors,
            recommendations=self._generate_recommendations(metrics, errors)
        )
