"""
백테스트 결과 평가 모듈
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
from ..utils.logger import setup_logger
from ..utils.config_loader import get_config

class StrategyEvaluator:
    """
    백테스트 결과를 평가하고 시각화하는 클래스
    """
    
    def __init__(
        self,
        results: Dict[str, Any],
        save_dir: str = 'results/analysis'
    ):
        """
        평가기 초기화
        
        Args:
            results (Dict[str, Any]): 백테스트 결과
            save_dir (str): 결과 저장 디렉토리
        """
        self.results = results
        self.save_dir = save_dir
        
        # 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 데이터 준비
        self.trades_df = pd.DataFrame(results['trades'])
        self.equity_curve = pd.Series(results['equity_curve'])
        
        # 로거 설정
        self.logger = setup_logger()
        self.logger.info("StrategyEvaluator initialized")
        
        # 결과 저장 경로
        self.config = get_config()['config']
        self.plots_dir = Path(self.config['backtest']['results']['save_path']) / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터프레임 변환
        self.trades_df = pd.DataFrame(self.results['trades'])
        self.equity_df = pd.DataFrame(self.results['equity_curve'])
        self.equity_df.set_index('timestamp', inplace=True)
        
        self.logger.info("전략 평가기 초기화 완료")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        전략 성과 평가
        
        Returns:
            Dict[str, Any]: 평가 결과
        """
        try:
            self.logger.info("Starting strategy evaluation")
            
            # 성과 지표 계산
            performance_metrics = self._calculate_performance_metrics()
            
            # 거래 통계 계산
            trade_statistics = self._calculate_trade_statistics()
            
            # 리스크 지표 계산
            risk_metrics = self._calculate_risk_metrics()
            
            # 결과 저장
            evaluation_results = {
                'performance_metrics': performance_metrics,
                'trade_statistics': trade_statistics,
                'risk_metrics': risk_metrics
            }
            
            self._save_evaluation_results(evaluation_results)
            
            self.logger.info("Strategy evaluation completed")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating strategy: {str(e)}")
            raise
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """
        성과 지표 계산
        
        Returns:
            Dict[str, float]: 성과 지표
        """
        try:
            # 기본 성과 지표
            total_return = self.results['summary']['total_return']
            win_rate = self.results['summary']['win_rate']
            sharpe_ratio = self.results['summary']['sharpe_ratio']
            
            # 수익-손실 비율
            profits = self.trades_df[self.trades_df['pnl'] > 0]['pnl']
            losses = self.trades_df[self.trades_df['pnl'] < 0]['pnl']
            profit_loss_ratio = abs(profits.mean() / losses.mean()) if len(losses) > 0 else float('inf')
            
            # 연간 수익률
            if len(self.trades_df) > 0:
                start_date = self.trades_df['entry_time'].min()
                end_date = self.trades_df['exit_time'].max()
                days = (end_date - start_date).days
                annual_return = (1 + total_return/100) ** (365/days) - 1 if days > 0 else 0
            else:
                annual_return = 0
            
            # 수익률 변동성
            returns = self.equity_curve.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            
            # 칼마 비율
            calmar_ratio = annual_return / self.results['summary']['max_drawdown'] if self.results['summary']['max_drawdown'] != 0 else float('inf')
            
            return {
                'total_return': total_return,
                'annual_return': annual_return * 100,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'volatility': volatility * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
    
    def _calculate_trade_statistics(self) -> Dict[str, Any]:
        """
        거래 통계 계산
        
        Returns:
            Dict[str, Any]: 거래 통계
        """
        try:
            # 거래 기간 계산
            self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
            self.trades_df['exit_time'] = pd.to_datetime(self.trades_df['exit_time'])
            self.trades_df['holding_period'] = (self.trades_df['exit_time'] - self.trades_df['entry_time'])
            
            # 거래 통계
            total_trades = len(self.trades_df)
            avg_holding_period = self.trades_df['holding_period'].mean()
            trades_per_month = total_trades / ((self.trades_df['exit_time'].max() - self.trades_df['entry_time'].min()).days / 30)
            
            # 포지션 타입별 통계
            long_trades = self.trades_df[self.trades_df['position_type'] == 'long']
            short_trades = self.trades_df[self.trades_df['position_type'] == 'short']
            
            # 수익성 통계
            profitable_trades = self.trades_df[self.trades_df['pnl'] > 0]
            avg_profit = profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0
            avg_loss = self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean() if len(self.trades_df[self.trades_df['pnl'] < 0]) > 0 else 0
            
            return {
                'total_trades': total_trades,
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'avg_holding_period': str(avg_holding_period),
                'trades_per_month': trades_per_month,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'best_trade': self.trades_df['pnl'].max(),
                'worst_trade': self.trades_df['pnl'].min()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade statistics: {str(e)}")
            raise
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """
        리스크 지표 계산
        
        Returns:
            Dict[str, float]: 리스크 지표
        """
        try:
            # 최대 낙폭 (MDD)
            max_drawdown = self.results['summary']['max_drawdown']
            
            # 변동성
            returns = self.equity_curve.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            
            # 베타 (시장 대비 변동성)
            market_returns = pd.Series([0.0001] * len(returns))  # 예시 시장 수익률
            beta = np.cov(returns, market_returns)[0,1] / np.var(market_returns) if len(returns) > 0 else 0
            
            # 알파 (시장 초과 수익률)
            alpha = returns.mean() * 252 - beta * market_returns.mean() * 252 if len(returns) > 0 else 0
            
            # 정보 비율
            tracking_error = np.std(returns - market_returns) * np.sqrt(252) if len(returns) > 0 else 0
            information_ratio = (alpha / tracking_error) if tracking_error != 0 else 0
            
            return {
                'max_drawdown': max_drawdown,
                'volatility': volatility * 100,
                'beta': beta,
                'alpha': alpha * 100,
                'information_ratio': information_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            raise
    
    def plot_results(self) -> None:
        """
        결과 시각화
        """
        try:
            self.logger.info("Generating visualization plots")
            
            # 스타일 설정
            plt.style.use('seaborn')
            sns.set_palette("husl")
            
            # 1. 자본금 곡선
            self._plot_equity_curve()
            
            # 2. 수익 분포
            self._plot_returns_distribution()
            
            # 3. 드로다운 차트
            self._plot_drawdown()
            
            # 4. 월별 수익률 히트맵
            self._plot_monthly_returns_heatmap()
            
            # 5. 거래 분석
            self._plot_trade_analysis()
            
            self.logger.info("Visualization plots generated")
            
        except Exception as e:
            self.logger.error(f"Error plotting results: {str(e)}")
            raise
    
    def _plot_equity_curve(self) -> None:
        """자본금 곡선 플롯"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve.values)
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'equity_curve.png'))
        plt.close()
    
    def _plot_returns_distribution(self) -> None:
        """수익률 분포 플롯"""
        plt.figure(figsize=(10, 6))
        returns = self.equity_curve.pct_change().dropna()
        sns.histplot(returns, bins=50)
        plt.title('Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'returns_distribution.png'))
        plt.close()
    
    def _plot_drawdown(self) -> None:
        """드로다운 차트 플롯"""
        plt.figure(figsize=(12, 6))
        rolling_max = self.equity_curve.expanding().max()
        drawdown = (rolling_max - self.equity_curve) / rolling_max * 100
        plt.plot(drawdown.index, drawdown.values)
        plt.title('Drawdown')
        plt.xlabel('Time')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'drawdown.png'))
        plt.close()
    
    def _plot_monthly_returns_heatmap(self) -> None:
        """월별 수익률 히트맵 플롯"""
        plt.figure(figsize=(12, 8))
        returns = self.equity_curve.pct_change()
        monthly_returns = returns.resample('M').agg(lambda x: (x + 1).prod() - 1)
        monthly_returns = monthly_returns.to_frame('returns')
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        
        pivot_table = monthly_returns.pivot_table(
            values='returns',
            index='year',
            columns='month',
            aggfunc='first'
        )
        
        sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='RdYlGn')
        plt.title('Monthly Returns Heatmap')
        plt.savefig(os.path.join(self.save_dir, 'monthly_returns_heatmap.png'))
        plt.close()
    
    def _plot_trade_analysis(self) -> None:
        """거래 분석 플롯"""
        # 1. 수익/손실 분포
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.trades_df, x='pnl', bins=50)
        plt.title('Trade PnL Distribution')
        plt.xlabel('PnL')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'trade_pnl_distribution.png'))
        plt.close()
        
        # 2. 누적 수익
        plt.figure(figsize=(12, 6))
        cumulative_pnl = self.trades_df['pnl'].cumsum()
        plt.plot(self.trades_df['exit_time'], cumulative_pnl)
        plt.title('Cumulative PnL')
        plt.xlabel('Time')
        plt.ylabel('Cumulative PnL')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'cumulative_pnl.png'))
        plt.close()
    
    def _save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """
        평가 결과 저장
        
        Args:
            results (Dict[str, Any]): 평가 결과
        """
        try:
            # 결과 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"evaluation_results_{timestamp}.json"
            filepath = os.path.join(self.save_dir, filename)
            
            # 결과 저장
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            
            self.logger.info(f"Evaluation results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {str(e)}")
            raise 