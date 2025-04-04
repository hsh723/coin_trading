"""
성과 분석 자동화 보고서 모듈
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import telegram
import os
import numpy as np

class PerformanceReporter:
    """성과 분석 자동화 보고서 클래스"""
    
    def __init__(self, db_manager):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.report_recipients = {
            'email': [],
            'telegram': []
        }
        
    def generate_weekly_report(self):
        """주간 성과 보고서 생성"""
        try:
            # 기간 설정
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # 성과 데이터 조회
            performance_data = self.db_manager.get_performance_data(
                start_date=start_date,
                end_date=end_date
            )
            
            # 보고서 생성
            report = self._generate_report_content(performance_data, "주간")
            
            # 보고서 전송
            self._send_report(report, "주간 성과 보고서")
            
        except Exception as e:
            self.logger.error(f"주간 보고서 생성 실패: {str(e)}")
            
    def generate_monthly_report(self):
        """월간 성과 보고서 생성"""
        try:
            # 기간 설정
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # 성과 데이터 조회
            performance_data = self.db_manager.get_performance_data(
                start_date=start_date,
                end_date=end_date
            )
            
            # 보고서 생성
            report = self._generate_report_content(performance_data, "월간")
            
            # 보고서 전송
            self._send_report(report, "월간 성과 보고서")
            
        except Exception as e:
            self.logger.error(f"월간 보고서 생성 실패: {str(e)}")
            
    def _generate_report_content(self, performance_data: List[Dict], period: str) -> str:
        """
        보고서 내용 생성
        
        Args:
            performance_data (List[Dict]): 성과 데이터
            period (str): 보고서 기간
            
        Returns:
            str: 보고서 내용
        """
        try:
            # 데이터프레임 변환
            df = pd.DataFrame(performance_data)
            
            # 기본 통계 계산
            total_trades = len(df)
            winning_trades = len(df[df['pnl'] > 0])
            losing_trades = len(df[df['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 수익률 계산
            total_pnl = df['pnl'].sum()
            avg_pnl = df['pnl'].mean()
            max_drawdown = self._calculate_max_drawdown(df['cumulative_pnl'])
            
            # 위험 조정 수익률
            sharpe_ratio = self._calculate_sharpe_ratio(df['pnl'])
            sortino_ratio = self._calculate_sortino_ratio(df['pnl'])
            
            # 전략별 성과
            strategy_performance = df.groupby('strategy').agg({
                'pnl': ['sum', 'mean', 'count'],
                'win_rate': 'mean'
            }).round(2)
            
            # 보고서 내용 생성
            report = f"""
{period} 성과 보고서
====================

1. 거래 요약
-----------
- 총 거래 수: {total_trades}
- 승리 거래: {winning_trades}
- 패배 거래: {losing_trades}
- 승률: {win_rate:.2%}

2. 수익률 분석
-------------
- 총 수익: {total_pnl:.2f}
- 평균 수익: {avg_pnl:.2f}
- 최대 손실폭: {max_drawdown:.2f}
- 샤프 비율: {sharpe_ratio:.2f}
- 소르티노 비율: {sortino_ratio:.2f}

3. 전략별 성과
-------------
{strategy_performance.to_string()}

4. 차트
------
- 누적 수익률 추이
- 일별 수익률 분포
- 전략별 수익률 비교
"""
            return report
            
        except Exception as e:
            self.logger.error(f"보고서 내용 생성 실패: {str(e)}")
            return ""
            
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """
        최대 손실폭 계산
        
        Args:
            cumulative_returns (pd.Series): 누적 수익률
            
        Returns:
            float: 최대 손실폭
        """
        try:
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            return drawdown.min()
            
        except Exception as e:
            self.logger.error(f"최대 손실폭 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        샤프 비율 계산
        
        Args:
            returns (pd.Series): 수익률
            risk_free_rate (float): 무위험 수익률
            
        Returns:
            float: 샤프 비율
        """
        try:
            excess_returns = returns - risk_free_rate / 252  # 일별 무위험 수익률
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
        except Exception as e:
            self.logger.error(f"샤프 비율 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        소르티노 비율 계산
        
        Args:
            returns (pd.Series): 수익률
            risk_free_rate (float): 무위험 수익률
            
        Returns:
            float: 소르티노 비율
        """
        try:
            excess_returns = returns - risk_free_rate / 252  # 일별 무위험 수익률
            downside_returns = excess_returns[excess_returns < 0]
            downside_std = downside_returns.std()
            return excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
        except Exception as e:
            self.logger.error(f"소르티노 비율 계산 실패: {str(e)}")
            return 0.0
            
    def _send_report(self, report: str, subject: str):
        """
        보고서 전송
        
        Args:
            report (str): 보고서 내용
            subject (str): 제목
        """
        try:
            # 이메일 전송
            if self.report_recipients['email']:
                self._send_email_report(report, subject)
                
            # 텔레그램 전송
            if self.report_recipients['telegram']:
                self._send_telegram_report(report, subject)
                
        except Exception as e:
            self.logger.error(f"보고서 전송 실패: {str(e)}")
            
    def _send_email_report(self, report: str, subject: str):
        """
        이메일로 보고서 전송
        
        Args:
            report (str): 보고서 내용
            subject (str): 제목
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = os.getenv('SMTP_USER')
            msg['To'] = ', '.join(self.report_recipients['email'])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(report, 'plain'))
            
            with smtplib.SMTP(os.getenv('SMTP_HOST'), int(os.getenv('SMTP_PORT'))) as server:
                server.starttls()
                server.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASSWORD'))
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"이메일 보고서 전송 실패: {str(e)}")
            
    def _send_telegram_report(self, report: str, subject: str):
        """
        텔레그램으로 보고서 전송
        
        Args:
            report (str): 보고서 내용
            subject (str): 제목
        """
        try:
            bot = telegram.Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
            
            for chat_id in self.report_recipients['telegram']:
                message = f"*{subject}*\n\n{report}"
                bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            self.logger.error(f"텔레그램 보고서 전송 실패: {str(e)}")
            
    def set_report_recipients(self, recipients: Dict):
        """
        보고서 수신자 설정
        
        Args:
            recipients (Dict): 수신자 설정
        """
        self.report_recipients.update(recipients) 