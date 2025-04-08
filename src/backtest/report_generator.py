import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List
from datetime import datetime

class ReportGenerator:
    def __init__(self, results: Dict, strategy_name: str):
        self.results = results
        self.strategy_name = strategy_name
        
    def generate_html_report(self) -> str:
        """HTML 형식 리포트 생성"""
        return f"""
        <html>
            <body>
                <h1>백테스트 결과: {self.strategy_name}</h1>
                {self._generate_summary_section()}
                {self._generate_performance_charts()}
                {self._generate_trades_table()}
            </body>
        </html>
        """
