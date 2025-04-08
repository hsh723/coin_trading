import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go

class EmailReporter:
    def __init__(self, smtp_config: Dict):
        self.smtp_config = smtp_config
        self.templates = {}
        
    async def generate_report(self, data: Dict) -> str:
        """보고서 생성"""
        report = MIMEMultipart()
        report.attach(MIMEText(self._create_html_report(data), 'html'))
        return report

    def _create_html_report(self, data: Dict) -> str:
        """HTML 형식 보고서 생성"""
        performance_chart = self._create_performance_chart(data['performance'])
        return self.templates['daily_report'].format(
            date=pd.Timestamp.now().strftime('%Y-%m-%d'),
            performance_summary=data['summary'],
            chart=performance_chart
        )
