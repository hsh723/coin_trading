import pandas as pd
import plotly.graph_objects as go
from typing import Dict
from datetime import datetime
import jinja2

class ReportGenerator:
    def __init__(self, template_path: str):
        self.template_loader = jinja2.FileSystemLoader(searchpath="./templates")
        self.template_env = jinja2.Environment(loader=self.template_loader)
        
    async def generate_daily_report(self, performance_data: Dict) -> str:
        """일일 성과 보고서 생성"""
        template = self.template_env.get_template('daily_report.html')
        charts = self._generate_performance_charts(performance_data)
        
        return template.render(
            date=datetime.now().strftime('%Y-%m-%d'),
            performance=performance_data,
            charts=charts
        )
