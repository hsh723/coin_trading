import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.model_selection import ParameterGrid
from ..utils.logger import setup_logger
from ..backtest.backtester import Backtester
from ..analysis.performance_analyzer import PerformanceAnalyzer

class StrategyOptimizer:
    """전략 최적화 클래스"""
    
    def __init__(self):
        """전략 최적화 클래스 초기화"""
        self.logger = setup_logger('strategy_optimizer')
        self.backtester = Backtester()
        self.performance_analyzer = PerformanceAnalyzer()
        
    def optimize_parameters(self, data: pd.DataFrame, 
                          param_grid: Dict[str, List[Any]],
                          metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        파라미터 최적화
        
        Args:
            data (pd.DataFrame): 백테스팅 데이터
            param_grid (Dict[str, List[Any]]): 파라미터 그리드
            metric (str): 최적화 지표
            
        Returns:
            Dict[str, Any]: 최적 파라미터
        """
        try:
            best_params = None
            best_score = float('-inf')
            results = []
            
            # 파라미터 그리드 생성
            grid = ParameterGrid(param_grid)
            
            for params in grid:
                # 백테스팅 실행
                self.backtester.set_parameters(params)
                equity_curve = self.backtester.run(data)
                
                # 성과 분석
                report = self.performance_analyzer.analyze_equity_curve(equity_curve)
                
                # 점수 계산
                score = report.get(metric, float('-inf'))
                
                # 결과 저장
                results.append({
                    'params': params,
                    'score': score,
                    'report': report
                })
                
                # 최적 파라미터 업데이트
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            return {
                'best_params': best_params,
                'best_score': best_score,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"파라미터 최적화 실패: {str(e)}")
            return {}
            
    def optimize_timeframe(self, data: pd.DataFrame,
                          timeframes: List[str],
                          metric: str = 'sharpe_ratio') -> str:
        """
        시간 프레임 최적화
        
        Args:
            data (pd.DataFrame): 백테스팅 데이터
            timeframes (List[str]): 시간 프레임 목록
            metric (str): 최적화 지표
            
        Returns:
            str: 최적 시간 프레임
        """
        try:
            best_timeframe = None
            best_score = float('-inf')
            results = []
            
            for timeframe in timeframes:
                # 데이터 리샘플링
                resampled_data = self._resample_data(data, timeframe)
                
                # 백테스팅 실행
                equity_curve = self.backtester.run(resampled_data)
                
                # 성과 분석
                report = self.performance_analyzer.analyze_equity_curve(equity_curve)
                
                # 점수 계산
                score = report.get(metric, float('-inf'))
                
                # 결과 저장
                results.append({
                    'timeframe': timeframe,
                    'score': score,
                    'report': report
                })
                
                # 최적 시간 프레임 업데이트
                if score > best_score:
                    best_score = score
                    best_timeframe = timeframe
                    
            return best_timeframe
            
        except Exception as e:
            self.logger.error(f"시간 프레임 최적화 실패: {str(e)}")
            return None
            
    def optimize_symbols(self, data_dict: Dict[str, pd.DataFrame],
                        metric: str = 'sharpe_ratio') -> List[str]:
        """
        거래 심볼 최적화
        
        Args:
            data_dict (Dict[str, pd.DataFrame]): 심볼별 데이터
            metric (str): 최적화 지표
            
        Returns:
            List[str]: 최적 심볼 목록
        """
        try:
            results = []
            
            for symbol, data in data_dict.items():
                # 백테스팅 실행
                equity_curve = self.backtester.run(data)
                
                # 성과 분석
                report = self.performance_analyzer.analyze_equity_curve(equity_curve)
                
                # 점수 계산
                score = report.get(metric, float('-inf'))
                
                # 결과 저장
                results.append({
                    'symbol': symbol,
                    'score': score,
                    'report': report
                })
                
            # 점수 기준 정렬
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # 상위 3개 심볼 선택
            best_symbols = [r['symbol'] for r in results[:3]]
            
            return best_symbols
            
        except Exception as e:
            self.logger.error(f"거래 심볼 최적화 실패: {str(e)}")
            return []
            
    def optimize_risk_parameters(self, data: pd.DataFrame,
                               risk_params: Dict[str, List[Any]],
                               metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        리스크 파라미터 최적화
        
        Args:
            data (pd.DataFrame): 백테스팅 데이터
            risk_params (Dict[str, List[Any]]): 리스크 파라미터 그리드
            metric (str): 최적화 지표
            
        Returns:
            Dict[str, Any]: 최적 리스크 파라미터
        """
        try:
            best_params = None
            best_score = float('-inf')
            results = []
            
            # 파라미터 그리드 생성
            grid = ParameterGrid(risk_params)
            
            for params in grid:
                # 백테스팅 실행 (리스크 파라미터 적용)
                self.backtester.set_risk_parameters(params)
                equity_curve = self.backtester.run(data)
                
                # 성과 분석
                report = self.performance_analyzer.analyze_equity_curve(equity_curve)
                
                # 점수 계산
                score = report.get(metric, float('-inf'))
                
                # 결과 저장
                results.append({
                    'params': params,
                    'score': score,
                    'report': report
                })
                
                # 최적 파라미터 업데이트
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            return {
                'best_params': best_params,
                'best_score': best_score,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"리스크 파라미터 최적화 실패: {str(e)}")
            return {}
            
    def optimize_entry_exit_rules(self, data: pd.DataFrame,
                                rules: List[Dict[str, Any]],
                                metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        진입/청산 규칙 최적화
        
        Args:
            data (pd.DataFrame): 백테스팅 데이터
            rules (List[Dict[str, Any]]): 규칙 목록
            metric (str): 최적화 지표
            
        Returns:
            Dict[str, Any]: 최적 규칙
        """
        try:
            best_rule = None
            best_score = float('-inf')
            results = []
            
            for rule in rules:
                # 백테스팅 실행 (규칙 적용)
                self.backtester.set_entry_exit_rules(rule)
                equity_curve = self.backtester.run(data)
                
                # 성과 분석
                report = self.performance_analyzer.analyze_equity_curve(equity_curve)
                
                # 점수 계산
                score = report.get(metric, float('-inf'))
                
                # 결과 저장
                results.append({
                    'rule': rule,
                    'score': score,
                    'report': report
                })
                
                # 최적 규칙 업데이트
                if score > best_score:
                    best_score = score
                    best_rule = rule
                    
            return {
                'best_rule': best_rule,
                'best_score': best_score,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"진입/청산 규칙 최적화 실패: {str(e)}")
            return {}
            
    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        데이터 리샘플링
        
        Args:
            data (pd.DataFrame): 원본 데이터
            timeframe (str): 시간 프레임
            
        Returns:
            pd.DataFrame: 리샘플링된 데이터
        """
        try:
            # 시간 프레임 변환
            timeframe_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '1h': '1H',
                '4h': '4H',
                '1d': '1D'
            }
            
            # 데이터 리샘플링
            resampled = data.resample(timeframe_map[timeframe]).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"데이터 리샘플링 실패: {str(e)}")
            return data 