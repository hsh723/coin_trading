"""
보고서 생성 모듈

이 모듈은 트레이딩 시스템의 성과 보고서를 생성합니다.
주요 기능:
- PDF/HTML 형식의 보고서 생성
- 차트 및 그래프 시각화
- 텔레그램으로 보고서 전송
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import os
import json
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from src.analysis.performance import PerformanceMetrics, PositionMetrics

# 로거 설정
logger = logging.getLogger(__name__)

class ReportGenerator:
    """보고서 생성 클래스"""
    
    def __init__(self, config: Dict):
        """
        보고서 생성기 초기화
        
        Args:
            config (Dict): 설정 정보
        """
        self.config = config
        self.template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
        
        # 출력 디렉토리 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 템플릿 환경 설정
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        
        logger.info("ReportGenerator initialized")
    
    def generate_report(
        self,
        metrics: PerformanceMetrics,
        position_metrics: Dict[str, PositionMetrics],
        equity_curve: pd.Series,
        trades: List[Dict],
        format: str = 'html'
    ) -> str:
        """
        성과 보고서 생성
        
        Args:
            metrics (PerformanceMetrics): 성과 지표
            position_metrics (Dict[str, PositionMetrics]): 포지션별 성과 지표
            equity_curve (pd.Series): 자본금 곡선
            trades (List[Dict]): 거래 내역
            format (str): 출력 형식 ('html' or 'pdf')
            
        Returns:
            str: 생성된 보고서 파일 경로
        """
        try:
            # 차트 생성
            charts = self._create_charts(equity_curve, trades)
            
            # 보고서 데이터 준비
            report_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics.__dict__,
                'position_metrics': {
                    symbol: metrics.__dict__
                    for symbol, metrics in position_metrics.items()
                },
                'charts': charts,
                'trades': trades
            }
            
            # HTML 템플릿 렌더링
            template = self.env.get_template('report.html')
            html_content = template.render(**report_data)
            
            # HTML 파일 저장
            html_path = os.path.join(
                self.output_dir,
                f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            )
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # PDF 변환 (요청된 경우)
            if format == 'pdf':
                pdf_path = html_path.replace('.html', '.pdf')
                HTML(html_path).write_pdf(pdf_path)
                return pdf_path
            
            return html_path
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {str(e)}")
            raise
    
    def generate_telegram_report(
        self,
        metrics: PerformanceMetrics,
        position_metrics: Dict[str, PositionMetrics]
    ) -> str:
        """
        텔레그램용 성과 보고서 생성
        
        Args:
            metrics (PerformanceMetrics): 성과 지표
            position_metrics (Dict[str, PositionMetrics]): 포지션별 성과 지표
            
        Returns:
            str: 보고서 내용
        """
        try:
            # 템플릿 렌더링
            template = self.env.get_template('telegram_report.md')
            report_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics.__dict__,
                'position_metrics': {
                    symbol: metrics.__dict__
                    for symbol, metrics in position_metrics.items()
                }
            }
            
            return template.render(**report_data)
            
        except Exception as e:
            logger.error(f"텔레그램 보고서 생성 중 오류 발생: {str(e)}")
            raise
    
    def _create_charts(self, equity_curve: pd.Series, trades: List[Dict]) -> Dict:
        """
        차트 생성
        
        Args:
            equity_curve (pd.Series): 자본금 곡선
            trades (List[Dict]): 거래 내역
            
        Returns:
            Dict: 차트 데이터
        """
        try:
            charts = {}
            
            # 자본금 곡선 차트
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Equity'
            ))
            fig.update_layout(
                title='자본금 곡선',
                xaxis_title='날짜',
                yaxis_title='자본금',
                template='plotly_dark'
            )
            charts['equity_curve'] = fig.to_json()
            
            # 수익률 분포 차트
            returns = pd.Series([t['pnl'] for t in trades]).pct_change()
            fig = px.histogram(
                returns,
                title='수익률 분포',
                template='plotly_dark'
            )
            charts['returns_distribution'] = fig.to_json()
            
            # 일별 수익률 차트
            daily_returns = returns.resample('D').sum()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily_returns.index,
                y=daily_returns.values,
                name='Daily Returns'
            ))
            fig.update_layout(
                title='일별 수익률',
                xaxis_title='날짜',
                yaxis_title='수익률',
                template='plotly_dark'
            )
            charts['daily_returns'] = fig.to_json()
            
            # 낙폭 차트
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                mode='lines',
                name='Drawdown',
                fill='tozeroy'
            ))
            fig.update_layout(
                title='낙폭',
                xaxis_title='날짜',
                yaxis_title='낙폭',
                template='plotly_dark'
            )
            charts['drawdown'] = fig.to_json()
            
            return charts
            
        except Exception as e:
            logger.error(f"차트 생성 중 오류 발생: {str(e)}")
            raise 