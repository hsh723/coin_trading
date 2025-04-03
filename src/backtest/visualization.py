"""
백테스트 결과 시각화 모듈
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.utils.logger import setup_logger

logger = setup_logger()

class BacktestVisualizer:
    def __init__(self, results_dir: str = 'data/backtest_results'):
        """
        백테스트 시각화기 초기화
        
        Args:
            results_dir (str): 결과 저장 디렉토리
        """
        self.results_dir = results_dir
        self.logger = setup_logger()
        
        # 결과 디렉토리 생성
        os.makedirs(results_dir, exist_ok=True)
        
    def save_results(self, results: Dict[str, Any], symbol: str, timeframe: str):
        """
        백테스트 결과 저장
        
        Args:
            results (Dict[str, Any]): 백테스트 결과
            symbol (str): 거래 심볼
            timeframe (str): 시간대
        """
        try:
            # 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timeframe}_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            # 결과 저장
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
                
            self.logger.info(f"백테스트 결과 저장 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"백테스트 결과 저장 실패: {str(e)}")
            raise
            
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        백테스트 결과 로드
        
        Args:
            filepath (str): 결과 파일 경로
            
        Returns:
            Dict[str, Any]: 백테스트 결과
        """
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            self.logger.error(f"백테스트 결과 로드 실패: {str(e)}")
            raise
            
    def plot_equity_curve(self, results: Dict[str, Any], symbol: str, timeframe: str) -> str:
        """
        자본금 곡선 플롯 생성
        
        Args:
            results (Dict[str, Any]): 백테스트 결과
            symbol (str): 거래 심볼
            timeframe (str): 시간대
            
        Returns:
            str: 플롯 파일 경로
        """
        try:
            # 자본금 곡선 데이터 생성
            equity_curve = pd.Series(results['equity_curve'])
            
            # 플롯 생성
            fig = go.Figure()
            
            # 자본금 곡선
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    name='자본금',
                    line=dict(color='blue')
                )
            )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f"자본금 곡선 ({symbol} {timeframe})",
                xaxis_title="시간",
                yaxis_title="자본금 (USDT)",
                showlegend=True
            )
            
            # 파일 저장
            filename = f"equity_curve_{symbol}_{timeframe}.html"
            filepath = os.path.join(self.results_dir, filename)
            fig.write_html(filepath)
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"자본금 곡선 플롯 생성 실패: {str(e)}")
            raise
            
    def plot_drawdown(self, results: Dict[str, Any], symbol: str, timeframe: str) -> str:
        """
        낙폭 플롯 생성
        
        Args:
            results (Dict[str, Any]): 백테스트 결과
            symbol (str): 거래 심볼
            timeframe (str): 시간대
            
        Returns:
            str: 플롯 파일 경로
        """
        try:
            # 자본금 곡선 데이터 생성
            equity_curve = pd.Series(results['equity_curve'])
            
            # 낙폭 계산
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max * 100
            
            # 플롯 생성
            fig = go.Figure()
            
            # 낙폭 곡선
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name='낙폭',
                    line=dict(color='red')
                )
            )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f"낙폭 분석 ({symbol} {timeframe})",
                xaxis_title="시간",
                yaxis_title="낙폭 (%)",
                showlegend=True
            )
            
            # 파일 저장
            filename = f"drawdown_{symbol}_{timeframe}.html"
            filepath = os.path.join(self.results_dir, filename)
            fig.write_html(filepath)
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"낙폭 플롯 생성 실패: {str(e)}")
            raise
            
    def plot_trade_distribution(self, results: Dict[str, Any], symbol: str, timeframe: str) -> str:
        """
        거래 분포 플롯 생성
        
        Args:
            results (Dict[str, Any]): 백테스트 결과
            symbol (str): 거래 심볼
            timeframe (str): 시간대
            
        Returns:
            str: 플롯 파일 경로
        """
        try:
            # 거래 데이터 추출
            trades = pd.DataFrame(results['trades'])
            
            # 수익/손실 분포
            pnl_distribution = trades['pnl'].value_counts().sort_index()
            
            # 플롯 생성
            fig = go.Figure()
            
            # 수익/손실 분포
            fig.add_trace(
                go.Bar(
                    x=pnl_distribution.index,
                    y=pnl_distribution.values,
                    name='거래 분포'
                )
            )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f"거래 분포 ({symbol} {timeframe})",
                xaxis_title="손익 (USDT)",
                yaxis_title="거래 수",
                showlegend=True
            )
            
            # 파일 저장
            filename = f"trade_distribution_{symbol}_{timeframe}.html"
            filepath = os.path.join(self.results_dir, filename)
            fig.write_html(filepath)
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"거래 분포 플롯 생성 실패: {str(e)}")
            raise
            
    def generate_report(self, results: Dict[str, Any], symbol: str, timeframe: str) -> str:
        """
        백테스트 리포트 생성
        
        Args:
            results (Dict[str, Any]): 백테스트 결과
            symbol (str): 거래 심볼
            timeframe (str): 시간대
            
        Returns:
            str: 리포트 파일 경로
        """
        try:
            # 결과 저장
            results_file = self.save_results(results, symbol, timeframe)
            
            # 플롯 생성
            equity_plot = self.plot_equity_curve(results, symbol, timeframe)
            drawdown_plot = self.plot_drawdown(results, symbol, timeframe)
            distribution_plot = self.plot_trade_distribution(results, symbol, timeframe)
            
            # 리포트 생성
            report = f"""# 백테스트 리포트 ({symbol} {timeframe})

## 성과 지표
- 총 수익률: {results['total_return']*100:.2f}%
- 승률: {results['win_rate']*100:.2f}%
- 최대 낙폭: {results['max_drawdown']*100:.2f}%
- 샤프 비율: {results['sharpe_ratio']:.2f}
- 총 거래 수: {results['total_trades']}
- 평균 수익률: {results['avg_return']*100:.2f}%
- 수익률 표준편차: {results['return_std']*100:.2f}%

## 시각화
- [자본금 곡선]({os.path.basename(equity_plot)})
- [낙폭 분석]({os.path.basename(drawdown_plot)})
- [거래 분포]({os.path.basename(distribution_plot)})

## 상세 결과
- [전체 결과 데이터]({os.path.basename(results_file)})
"""
            
            # 리포트 저장
            filename = f"report_{symbol}_{timeframe}.md"
            filepath = os.path.join(self.results_dir, filename)
            with open(filepath, 'w') as f:
                f.write(report)
                
            self.logger.info(f"백테스트 리포트 생성 완료: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"백테스트 리포트 생성 실패: {str(e)}")
            raise 