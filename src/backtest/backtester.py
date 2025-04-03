import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from ..analysis.technical_analyzer import TechnicalAnalyzer
from ..strategy.integrated_strategy import IntegratedStrategy
from ..risk.risk_manager import RiskManager
from ..utils.logger import setup_logger

class Backtester:
    """백테스팅 클래스"""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        백테스터 초기화
        
        Args:
            initial_capital (float): 초기 자본금
        """
        self.logger = setup_logger('backtester')
        self.initial_capital = initial_capital
        self.technical_analyzer = TechnicalAnalyzer()
        self.strategy = IntegratedStrategy()
        self.risk_manager = RiskManager(initial_capital=initial_capital)
        
    def run(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        백테스팅 실행
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            params (Dict[str, Any]): 백테스팅 파라미터
            
        Returns:
            Dict[str, Any]: 백테스팅 결과
        """
        try:
            # 데이터 전처리
            data = self._preprocess_data(data)
            
            # 기술적 지표 계산
            data = self.technical_analyzer.calculate_indicators(data)
            
            # 백테스팅 결과 초기화
            results = {
                'trades': [],
                'equity_curve': [],
                'metrics': {},
                'positions': []
            }
            
            # 현재 포지션
            current_position = None
            
            # 자본금
            capital = self.initial_capital
            
            # 거래 내역
            trades = []
            
            # 일별 수익률
            daily_returns = []
            
            # 백테스팅 루프
            for i in range(len(data)):
                current_data = data.iloc[i]
                
                # 시장 데이터
                market_data = {
                    'ohlcv': data.iloc[:i+1],
                    'current_price': current_data['close'],
                    'volatility': self.technical_analyzer.calculate_volatility(data.iloc[:i+1]),
                    'trend_strength': self.technical_analyzer.calculate_trend_strength(data.iloc[:i+1])
                }
                
                # 리스크 파라미터 조정
                self.risk_manager.adjust_risk_parameters(market_data['volatility'])
                
                # 거래 신호 생성
                signal = self.strategy.generate_signal(market_data)
                
                # 포지션 관리
                if signal and signal['signal'] != 'neutral':
                    if not current_position and signal['signal'] == 'buy':
                        # 매수 진입
                        entry_price = current_data['close']
                        stop_loss = self.strategy.calculate_stop_loss(
                            entry_price,
                            current_data['atr']
                        )
                        position_size = self.risk_manager.calculate_position_size(
                            entry_price,
                            stop_loss
                        )
                        
                        current_position = {
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': self.strategy.calculate_take_profit(
                                entry_price,
                                stop_loss
                            ),
                            'size': position_size,
                            'entry_time': current_data.name
                        }
                        
                        trades.append({
                            'entry_time': current_data.name,
                            'entry_price': entry_price,
                            'side': 'buy',
                            'size': position_size
                        })
                        
                    elif current_position and signal['signal'] == 'sell':
                        # 매도 청산
                        exit_price = current_data['close']
                        pnl = (exit_price - current_position['entry_price']) * current_position['size']
                        capital += pnl
                        
                        trades[-1].update({
                            'exit_time': current_data.name,
                            'exit_price': exit_price,
                            'pnl': pnl
                        })
                        
                        current_position = None
                        
                # 손절/이익 실현 체크
                if current_position:
                    if current_data['low'] <= current_position['stop_loss']:
                        # 손절
                        pnl = (current_position['stop_loss'] - current_position['entry_price']) * current_position['size']
                        capital += pnl
                        
                        trades[-1].update({
                            'exit_time': current_data.name,
                            'exit_price': current_position['stop_loss'],
                            'pnl': pnl,
                            'exit_type': 'stop_loss'
                        })
                        
                        current_position = None
                        
                    elif current_data['high'] >= current_position['take_profit']:
                        # 이익 실현
                        pnl = (current_position['take_profit'] - current_position['entry_price']) * current_position['size']
                        capital += pnl
                        
                        trades[-1].update({
                            'exit_time': current_data.name,
                            'exit_price': current_position['take_profit'],
                            'pnl': pnl,
                            'exit_type': 'take_profit'
                        })
                        
                        current_position = None
                
                # 자본금 곡선 업데이트
                results['equity_curve'].append({
                    'timestamp': current_data.name,
                    'equity': capital
                })
                
                # 일별 수익률 계산
                if i > 0:
                    daily_return = (capital / results['equity_curve'][-2]['equity']) - 1
                    daily_returns.append(daily_return)
            
            # 성과 지표 계산
            results['trades'] = trades
            results['metrics'] = self._calculate_metrics(
                pd.DataFrame(results['equity_curve']),
                pd.Series(daily_returns),
                trades
            )
            
            self.logger.info("백테스팅 완료")
            return results
            
        except Exception as e:
            self.logger.error(f"백테스팅 실패: {str(e)}")
            return {}
            
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 전처리
        
        Args:
            data (pd.DataFrame): 원본 데이터
            
        Returns:
            pd.DataFrame: 전처리된 데이터
        """
        try:
            # 인덱스가 datetime이 아니면 변환
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
                
            # 결측치 처리
            data = data.fillna(method='ffill')
            
            # 데이터 정렬
            data = data.sort_index()
            
            return data
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 실패: {str(e)}")
            return data
            
    def _calculate_metrics(self, equity_curve: pd.DataFrame, daily_returns: pd.Series, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        성과 지표 계산
        
        Args:
            equity_curve (pd.DataFrame): 자본금 곡선
            daily_returns (pd.Series): 일별 수익률
            trades (List[Dict[str, Any]]): 거래 내역
            
        Returns:
            Dict[str, float]: 성과 지표
        """
        try:
            # 총 수익률
            total_return = (equity_curve['equity'].iloc[-1] / self.initial_capital) - 1
            
            # 연간 수익률
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            annual_return = (1 + total_return) ** (365 / days) - 1
            
            # 최대 낙폭
            equity_curve['peak'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['peak']) / equity_curve['peak']
            max_drawdown = equity_curve['drawdown'].min()
            
            # 샤프 비율
            risk_free_rate = 0.02  # 연간 무위험 수익률 2%
            daily_risk_free = (1 + risk_free_rate) ** (1/365) - 1
            excess_returns = daily_returns - daily_risk_free
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # 승률
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            total_trades = len(trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 평균 수익/손실
            profits = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
            losses = [t['pnl'] for t in trades if t.get('pnl', 0) < 0]
            avg_profit = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # 수익/손실 비율
            profit_factor = abs(sum(profits) / sum(losses)) if losses else float('inf')
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            self.logger.error(f"성과 지표 계산 실패: {str(e)}")
            return {}
            
    def optimize_parameters(self, data: pd.DataFrame, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        파라미터 최적화
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            param_grid (Dict[str, List[Any]]): 파라미터 그리드
            
        Returns:
            Dict[str, Any]: 최적 파라미터
        """
        try:
            best_params = None
            best_sharpe = float('-inf')
            
            # 파라미터 조합 생성
            from itertools import product
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(product(*param_values))
            
            # 각 조합에 대해 백테스팅 실행
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                
                # 전략 파라미터 업데이트
                self.strategy.update_parameters(param_dict)
                
                # 백테스팅 실행
                results = self.run(data, param_dict)
                
                if results and 'metrics' in results:
                    sharpe = results['metrics'].get('sharpe_ratio', float('-inf'))
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = param_dict
            
            self.logger.info(f"파라미터 최적화 완료: {best_params}")
            return best_params
            
        except Exception as e:
            self.logger.error(f"파라미터 최적화 실패: {str(e)}")
            return {}
            
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        백테스팅 리포트 생성
        
        Args:
            results (Dict[str, Any]): 백테스팅 결과
            
        Returns:
            str: 리포트 내용
        """
        try:
            metrics = results.get('metrics', {})
            
            report = f"""
            백테스팅 리포트
            ===============
            
            성과 지표
            ---------
            총 수익률: {metrics.get('total_return', 0):.2%}
            연간 수익률: {metrics.get('annual_return', 0):.2%}
            최대 낙폭: {metrics.get('max_drawdown', 0):.2%}
            샤프 비율: {metrics.get('sharpe_ratio', 0):.2f}
            
            거래 통계
            ---------
            총 거래 횟수: {metrics.get('total_trades', 0)}
            승률: {metrics.get('win_rate', 0):.2%}
            평균 수익: {metrics.get('avg_profit', 0):.2f}
            평균 손실: {metrics.get('avg_loss', 0):.2f}
            수익/손실 비율: {metrics.get('profit_factor', 0):.2f}
            """
            
            return report
            
        except Exception as e:
            self.logger.error(f"리포트 생성 실패: {str(e)}")
            return "" 