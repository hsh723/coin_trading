"""
포트폴리오 관리 모듈
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from src.database.database import DatabaseManager
from src.analysis.technical_analyzer import TechnicalAnalyzer

class PortfolioManager:
    """포트폴리오 관리자 클래스"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.technical_analyzer = TechnicalAnalyzer()
        self.strategies = {}
        self.weights = {}
        self.performance_history = {}
        
    def add_strategy(self, 
                    strategy_id: str,
                    strategy_class: type,
                    initial_weight: float = 0.0):
        """
        전략 추가
        
        Args:
            strategy_id (str): 전략 ID
            strategy_class (type): 전략 클래스
            initial_weight (float): 초기 가중치
        """
        try:
            self.strategies[strategy_id] = strategy_class()
            self.weights[strategy_id] = initial_weight
            self.performance_history[strategy_id] = []
            self.logger.info(f"전략 추가 완료: {strategy_id}")
            
        except Exception as e:
            self.logger.error(f"전략 추가 실패: {str(e)}")
            raise
            
    def remove_strategy(self, strategy_id: str):
        """
        전략 제거
        
        Args:
            strategy_id (str): 전략 ID
        """
        try:
            if strategy_id in self.strategies:
                del self.strategies[strategy_id]
                del self.weights[strategy_id]
                del self.performance_history[strategy_id]
                self.logger.info(f"전략 제거 완료: {strategy_id}")
                
        except Exception as e:
            self.logger.error(f"전략 제거 실패: {str(e)}")
            raise
            
    def update_weights(self, 
                      lookback_days: int = 30,
                      min_weight: float = 0.1,
                      max_weight: float = 0.5):
        """
        전략 가중치 업데이트
        
        Args:
            lookback_days (int): 성과 분석 기간
            min_weight (float): 최소 가중치
            max_weight (float): 최대 가중치
        """
        try:
            # 전략별 성과 계산
            performances = {}
            for strategy_id in self.strategies:
                performance = self._calculate_strategy_performance(
                    strategy_id,
                    lookback_days
                )
                if performance is not None:
                    performances[strategy_id] = performance
                    
            if not performances:
                self.logger.warning("성과 데이터가 없어 가중치를 업데이트할 수 없습니다.")
                return
                
            # 성과 기반 가중치 계산
            total_performance = sum(performances.values())
            if total_performance > 0:
                for strategy_id in performances:
                    raw_weight = performances[strategy_id] / total_performance
                    # 가중치 범위 제한
                    self.weights[strategy_id] = np.clip(
                        raw_weight,
                        min_weight,
                        max_weight
                    )
                    
            # 가중치 정규화
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for strategy_id in self.weights:
                    self.weights[strategy_id] /= total_weight
                    
            self.logger.info(f"전략 가중치 업데이트 완료: {self.weights}")
            
        except Exception as e:
            self.logger.error(f"전략 가중치 업데이트 실패: {str(e)}")
            raise
            
    def _calculate_strategy_performance(self,
                                      strategy_id: str,
                                      lookback_days: int) -> Optional[float]:
        """
        전략 성과 계산
        
        Args:
            strategy_id (str): 전략 ID
            lookback_days (int): 성과 분석 기간
            
        Returns:
            Optional[float]: 성과 점수
        """
        try:
            # 거래 기록 조회
            trades = self.db_manager.get_trades(
                strategy_id=strategy_id,
                start_time=datetime.now() - timedelta(days=lookback_days)
            )
            
            if len(trades) < 5:
                self.logger.warning(f"전략 {strategy_id}의 거래 데이터가 부족합니다.")
                return None
                
            # 성과 지표 계산
            total_pnl = sum(trade['pnl'] for trade in trades)
            win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades)
            profit_factor = self._calculate_profit_factor(trades)
            sharpe_ratio = self._calculate_sharpe_ratio(trades)
            
            # 종합 성과 점수 계산
            performance_score = (
                total_pnl * 0.4 +
                win_rate * 0.3 +
                profit_factor * 0.2 +
                sharpe_ratio * 0.1
            )
            
            # 성과 기록 업데이트
            self.performance_history[strategy_id].append({
                'timestamp': datetime.now(),
                'score': performance_score,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio
            })
            
            return performance_score
            
        except Exception as e:
            self.logger.error(f"전략 성과 계산 실패: {str(e)}")
            return None
            
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """
        수익률 계산
        
        Args:
            trades (List[Dict]): 거래 기록
            
        Returns:
            float: 수익률
        """
        try:
            profits = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
            losses = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
            return profits / losses if losses > 0 else float('inf')
            
        except Exception as e:
            self.logger.error(f"수익률 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """
        샤프 비율 계산
        
        Args:
            trades (List[Dict]): 거래 기록
            
        Returns:
            float: 샤프 비율
        """
        try:
            returns = [trade['pnl'] / trade['entry_price'] for trade in trades]
            if len(returns) < 2:
                return 0.0
                
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            return avg_return / std_return if std_return > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"샤프 비율 계산 실패: {str(e)}")
            return 0.0
            
    def get_position_size(self,
                         strategy_id: str,
                         symbol: str,
                         available_capital: float) -> float:
        """
        포지션 크기 계산
        
        Args:
            strategy_id (str): 전략 ID
            symbol (str): 심볼
            available_capital (float): 가용 자본
            
        Returns:
            float: 포지션 크기
        """
        try:
            if strategy_id not in self.weights:
                self.logger.warning(f"전략 {strategy_id}의 가중치가 없습니다.")
                return 0.0
                
            # 전략 가중치 기반 포지션 크기 계산
            position_size = available_capital * self.weights[strategy_id]
            
            # 리스크 관리 제한 적용
            risk_params = self.db_manager.get_risk_parameters()
            max_position_size = available_capital * risk_params['max_position_size']
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {str(e)}")
            return 0.0
            
    def get_portfolio_status(self) -> Dict:
        """
        포트폴리오 상태 조회
        
        Returns:
            Dict: 포트폴리오 상태
        """
        try:
            status = {
                'strategies': {},
                'total_weight': sum(self.weights.values()),
                'total_performance': 0.0
            }
            
            for strategy_id in self.strategies:
                if strategy_id in self.performance_history:
                    recent_performance = self.performance_history[strategy_id][-1]
                    status['strategies'][strategy_id] = {
                        'weight': self.weights[strategy_id],
                        'performance': recent_performance['score'],
                        'total_pnl': recent_performance['total_pnl'],
                        'win_rate': recent_performance['win_rate'],
                        'profit_factor': recent_performance['profit_factor'],
                        'sharpe_ratio': recent_performance['sharpe_ratio']
                    }
                    status['total_performance'] += (
                        recent_performance['score'] * self.weights[strategy_id]
                    )
                    
            return status
            
        except Exception as e:
            self.logger.error(f"포트폴리오 상태 조회 실패: {str(e)}")
            return {} 