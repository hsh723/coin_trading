import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List

class PerformanceVisualizer:
    def __init__(self, theme: str = 'dark'):
        self.theme = theme
        self.color_scheme = {
            'profit': 'rgba(0, 255, 0, 0.7)',
            'loss': 'rgba(255, 0, 0, 0.7)',
            'line': 'rgba(0, 150, 255, 0.8)'
        }
        
    def create_performance_dashboard(self, results: Dict) -> List[go.Figure]:
        """백테스트 결과 대시보드 생성"""
        figures = []
        
        # 수익률 차트
        figures.append(self._create_equity_curve(results['equity_curve']))
        
        # 월간 수익률 히트맵
        figures.append(self._create_monthly_heatmap(results['monthly_returns']))
        
        # 리스크 메트릭스
        figures.append(self._create_risk_metrics(results['risk_metrics']))
        
        return figures
