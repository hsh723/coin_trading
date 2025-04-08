"""
실시간 모니터링 대시보드 모듈
"""

import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os
from src.utils.logger import setup_logger
from src.utils.config_loader import get_config

logger = setup_logger()

class MonitoringDashboard:
    def __init__(
        self,
        config: Dict[str, Any],
        executor: Any,
        notifier: Any
    ):
        """
        모니터링 대시보드 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
            executor (Any): 주문 실행기
            notifier (Any): 알림 전송기
        """
        self.config = config
        self.executor = executor
        self.notifier = notifier
        self.logger = setup_logger()
        
        # 데이터 저장소
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        
        # 상태 변수
        self.is_running = False
        self.update_interval = config.get('dashboard', {}).get('update_interval', 60)  # 초
        
        # 결과 저장 경로
        self.results_dir = os.path.join('data', 'dashboard')
        os.makedirs(self.results_dir, exist_ok=True)
        
    async def start(self):
        """
        대시보드 시작
        """
        try:
            self.is_running = True
            
            # 초기 데이터 로드
            await self._load_initial_data()
            
            # 업데이트 루프 시작
            asyncio.create_task(self._update_loop())
            
            self.logger.info("모니터링 대시보드 시작")
            
        except Exception as e:
            self.logger.error(f"대시보드 시작 실패: {str(e)}")
            raise
            
    async def stop(self):
        """
        대시보드 종료
        """
        try:
            self.is_running = False
            
            # 최종 데이터 저장
            await self._save_data()
            
            self.logger.info("모니터링 대시보드 종료")
            
        except Exception as e:
            self.logger.error(f"대시보드 종료 실패: {str(e)}")
            raise
            
    async def _load_initial_data(self):
        """
        초기 데이터 로드
        """
        try:
            # 시장 데이터 로드
            for symbol in self.config['trading']['symbols']:
                self.market_data[symbol] = await self.executor.get_market_data(
                    symbol,
                    interval='1h',
                    limit=100
                )
                
            # 포지션 정보 로드
            self.positions = await self.executor.get_positions()
            
            # 자본금 정보 초기화
            self.equity_curve.append({
                'timestamp': datetime.now(),
                'equity': self.executor.initial_capital
            })
            
        except Exception as e:
            self.logger.error(f"초기 데이터 로드 실패: {str(e)}")
            raise
            
    async def _update_loop(self):
        """
        데이터 업데이트 루프
        """
        while self.is_running:
            try:
                # 데이터 업데이트
                await self._update_data()
                
                # 대시보드 생성
                await self._generate_dashboard()
                
                # 성능 리포트 생성
                await self._generate_performance_report()
                
                # 데이터 저장
                await self._save_data()
                
                # 대기
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"데이터 업데이트 실패: {str(e)}")
                await asyncio.sleep(60)  # 오류 발생 시 1분 대기
                
    async def _update_data(self):
        """
        데이터 업데이트
        """
        try:
            # 시장 데이터 업데이트
            for symbol in self.config['trading']['symbols']:
                new_data = await self.executor.get_market_data(
                    symbol,
                    interval='1h',
                    limit=1
                )
                
                if symbol not in self.market_data:
                    self.market_data[symbol] = new_data
                else:
                    self.market_data[symbol] = pd.concat([
                        self.market_data[symbol],
                        new_data
                    ]).tail(1000)  # 최근 1000개 데이터만 유지
                    
            # 포지션 정보 업데이트
            self.positions = await self.executor.get_positions()
            
            # 자본금 정보 업데이트
            current_equity = self.executor.initial_capital
            for position in self.positions.values():
                current_equity += position.get('pnl', 0)
                
            self.equity_curve.append({
                'timestamp': datetime.now(),
                'equity': current_equity
            })
            
        except Exception as e:
            self.logger.error(f"데이터 업데이트 실패: {str(e)}")
            raise
            
    async def _generate_dashboard(self):
        """
        대시보드 생성
        """
        try:
            # 서브플롯 생성
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    '자본금 곡선',
                    '포지션 분포',
                    '시장 가격',
                    '거래량',
                    '일간 수익률',
                    '포지션 손익'
                )
            )
            
            # 자본금 곡선
            equity_df = pd.DataFrame(self.equity_curve)
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['equity'],
                    name='자본금',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # 포지션 분포
            position_sizes = [
                position['size'] * position['current_price']
                for position in self.positions.values()
            ]
            position_symbols = list(self.positions.keys())
            
            fig.add_trace(
                go.Pie(
                    labels=position_symbols,
                    values=position_sizes,
                    name='포지션 분포'
                ),
                row=1, col=2
            )
            
            # 시장 가격
            for symbol in self.config['trading']['symbols']:
                data = self.market_data[symbol]
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data['close'],
                        name=symbol,
                        line=dict(width=1)
                    ),
                    row=2, col=1
                )
                
            # 거래량
            for symbol in self.config['trading']['symbols']:
                data = self.market_data[symbol]
                fig.add_trace(
                    go.Bar(
                        x=data['timestamp'],
                        y=data['volume'],
                        name=symbol,
                        opacity=0.5
                    ),
                    row=2, col=2
                )
                
            # 일간 수익률
            daily_returns = equity_df.set_index('timestamp')['equity'].pct_change()
            fig.add_trace(
                go.Bar(
                    x=daily_returns.index,
                    y=daily_returns.values,
                    name='일간 수익률'
                ),
                row=3, col=1
            )
            
            # 포지션 손익
            for symbol, position in self.positions.items():
                fig.add_trace(
                    go.Bar(
                        x=[symbol],
                        y=[position.get('pnl', 0)],
                        name=symbol
                    ),
                    row=3, col=2
                )
                
            # 레이아웃 설정
            fig.update_layout(
                height=1200,
                showlegend=True,
                title_text="실시간 트레이딩 대시보드",
                title_x=0.5
            )
            
            # 대시보드 저장
            dashboard_path = os.path.join(
                self.results_dir,
                f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            fig.write_html(dashboard_path)
            
        except Exception as e:
            self.logger.error(f"대시보드 생성 실패: {str(e)}")
            raise
            
    async def _generate_performance_report(self):
        """
        성능 리포트 생성
        """
        try:
            # 성능 지표 계산
            metrics = self._calculate_performance_metrics()
            
            # 리포트 생성
            report = (
                f"📊 성능 리포트 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n"
                f"💰 자본금 상태\n"
                f"- 현재 자본금: {metrics['current_equity']:.2f} USDT\n"
                f"- 총 수익률: {metrics['total_return']*100:.2f}%\n"
                f"- 일간 수익률: {metrics['daily_return']*100:.2f}%\n\n"
                f"📈 포지션 상태\n"
                f"- 오픈 포지션: {len(self.positions)}개\n"
                f"- 총 포지션 가치: {metrics['total_position_value']:.2f} USDT\n"
                f"- 미실현 손익: {metrics['unrealized_pnl']:.2f} USDT\n\n"
                f"📊 거래 활동\n"
                f"- 일간 거래 수: {metrics['daily_trades']}회\n"
                f"- 승률: {metrics['win_rate']*100:.2f}%\n"
                f"- 평균 수익률: {metrics['avg_return']*100:.2f}%\n\n"
                f"⚠️ 리스크 지표\n"
                f"- 일간 변동성: {metrics['daily_volatility']*100:.2f}%\n"
                f"- 최대 낙폭: {metrics['max_drawdown']*100:.2f}%\n"
                f"- 샤프 비율: {metrics['sharpe_ratio']:.2f}"
            )
            
            # 리포트 저장
            report_path = os.path.join(
                self.results_dir,
                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
                
            # 텔레그램으로 리포트 전송
            await self.notifier.send_message(report)
            
        except Exception as e:
            self.logger.error(f"성능 리포트 생성 실패: {str(e)}")
            raise
            
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """
        성능 지표 계산
        
        Returns:
            Dict[str, float]: 성능 지표
        """
        try:
            # 자본금 상태
            current_equity = self.equity_curve[-1]['equity']
            total_return = (current_equity - self.executor.initial_capital) / self.executor.initial_capital
            
            # 일간 수익률
            daily_returns = pd.DataFrame(self.equity_curve).set_index('timestamp')['equity'].pct_change()
            daily_return = daily_returns.iloc[-1] if not daily_returns.empty else 0
            
            # 포지션 상태
            total_position_value = sum(
                position['size'] * position['current_price']
                for position in self.positions.values()
            )
            unrealized_pnl = sum(
                position.get('pnl', 0)
                for position in self.positions.values()
            )
            
            # 거래 활동
            daily_trades = len([
                trade for trade in self.trades
                if trade['timestamp'].date() == datetime.now().date()
            ])
            
            # 승률 및 평균 수익률
            winning_trades = [trade for trade in self.trades if trade['pnl'] > 0]
            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
            avg_return = np.mean([trade['pnl'] for trade in self.trades]) if self.trades else 0
            
            # 리스크 지표
            daily_volatility = daily_returns.std() if not daily_returns.empty else 0
            
            # 최대 낙폭
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0
            
            # 샤프 비율
            risk_free_rate = 0.02 / 252  # 연간 2% 가정
            sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std() if not daily_returns.empty else 0
            
            return {
                'current_equity': current_equity,
                'total_return': total_return,
                'daily_return': daily_return,
                'total_position_value': total_position_value,
                'unrealized_pnl': unrealized_pnl,
                'daily_trades': daily_trades,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'daily_volatility': daily_volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            self.logger.error(f"성능 지표 계산 실패: {str(e)}")
            raise
            
    async def _save_data(self):
        """
        데이터 저장
        """
        try:
            # 시장 데이터 저장
            for symbol, data in self.market_data.items():
                data.to_csv(
                    os.path.join(self.results_dir, f"market_data_{symbol}.csv"),
                    index=False
                )
                
            # 포지션 정보 저장
            positions_df = pd.DataFrame(self.positions.values())
            if not positions_df.empty:
                positions_df.to_csv(
                    os.path.join(self.results_dir, "positions.csv"),
                    index=False
                )
                
            # 거래 내역 저장
            trades_df = pd.DataFrame(self.trades)
            if not trades_df.empty:
                trades_df.to_csv(
                    os.path.join(self.results_dir, "trades.csv"),
                    index=False
                )
                
            # 자본금 곡선 저장
            equity_df = pd.DataFrame(self.equity_curve)
            if not equity_df.empty:
                equity_df.to_csv(
                    os.path.join(self.results_dir, "equity_curve.csv"),
                    index=False
                )
                
        except Exception as e:
            self.logger.error(f"데이터 저장 실패: {str(e)}")
            raise 