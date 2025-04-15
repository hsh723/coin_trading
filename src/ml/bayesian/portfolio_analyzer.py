import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class PortfolioAnalyzer:
    """포트폴리오 분석 시스템"""
    
    def __init__(self,
                 config_path: str = "./config/portfolio_config.json",
                 log_dir: str = "./logs"):
        """
        포트폴리오 분석 시스템 초기화
        
        Args:
            config_path: 설정 파일 경로
            log_dir: 로그 디렉토리
        """
        self.config_path = config_path
        self.log_dir = log_dir
        
        # 로거 설정
        self.logger = logging.getLogger("portfolio_analyzer")
        
        # 분석 데이터
        self.returns = None
        self.weights = None
        self.portfolio_value = None
        
        # 설정 로드
        self.config = self._load_config()
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
            
    def update_data(self, returns: pd.DataFrame, weights: np.ndarray) -> None:
        """
        분석 데이터 업데이트
        
        Args:
            returns: 자산별 수익률 데이터프레임
            weights: 자산별 가중치 배열
        """
        try:
            self.returns = returns
            self.weights = weights
            self.portfolio_value = self._calculate_portfolio_value()
            self.logger.info("분석 데이터 업데이트 완료")
        except Exception as e:
            self.logger.error(f"데이터 업데이트 중 오류 발생: {e}")
            
    def _calculate_portfolio_value(self) -> pd.Series:
        """포트폴리오 가치 계산"""
        try:
            portfolio_returns = np.dot(self.returns, self.weights)
            portfolio_value = np.cumprod(1 + portfolio_returns)
            return pd.Series(portfolio_value, index=self.returns.index)
        except Exception as e:
            self.logger.error(f"포트폴리오 가치 계산 중 오류 발생: {e}")
            return None
            
    def analyze_returns(self) -> Dict[str, float]:
        """
        수익률 분석
        
        Returns:
            수익률 분석 결과 딕셔너리
        """
        try:
            portfolio_returns = np.dot(self.returns, self.weights)
            
            analysis = {
                "total_return": self._calculate_total_return(portfolio_returns),
                "annual_return": self._calculate_annual_return(portfolio_returns),
                "monthly_return": self._calculate_monthly_return(portfolio_returns),
                "daily_return": self._calculate_daily_return(portfolio_returns),
                "sharpe_ratio": self._calculate_sharpe_ratio(portfolio_returns),
                "sortino_ratio": self._calculate_sortino_ratio(portfolio_returns),
                "calmar_ratio": self._calculate_calmar_ratio(portfolio_returns),
                "win_rate": self._calculate_win_rate(portfolio_returns),
                "profit_factor": self._calculate_profit_factor(portfolio_returns),
                "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
                "recovery_factor": self._calculate_recovery_factor(portfolio_returns)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"수익률 분석 중 오류 발생: {e}")
            return {}
            
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """총 수익률 계산"""
        try:
            return np.prod(1 + returns) - 1
        except Exception as e:
            self.logger.error(f"총 수익률 계산 중 오류 발생: {e}")
            return None
            
    def _calculate_annual_return(self, returns: np.ndarray) -> float:
        """연간 수익률 계산"""
        try:
            days = len(returns)
            annual_factor = 252 / days
            return (1 + np.mean(returns)) ** annual_factor - 1
        except Exception as e:
            self.logger.error(f"연간 수익률 계산 중 오류 발생: {e}")
            return None
            
    def _calculate_monthly_return(self, returns: np.ndarray) -> float:
        """월간 수익률 계산"""
        try:
            days = len(returns)
            monthly_factor = 21 / days
            return (1 + np.mean(returns)) ** monthly_factor - 1
        except Exception as e:
            self.logger.error(f"월간 수익률 계산 중 오류 발생: {e}")
            return None
            
    def _calculate_daily_return(self, returns: np.ndarray) -> float:
        """일간 수익률 계산"""
        try:
            return np.mean(returns)
        except Exception as e:
            self.logger.error(f"일간 수익률 계산 중 오류 발생: {e}")
            return None
            
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """샤프 비율 계산"""
        try:
            excess_returns = returns - self.risk_free_rate/252
            return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        except Exception as e:
            self.logger.error(f"샤프 비율 계산 중 오류 발생: {e}")
            return None
            
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """소르티노 비율 계산"""
        try:
            excess_returns = returns - self.risk_free_rate/252
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return np.inf
            downside_std = np.std(downside_returns)
            return np.mean(excess_returns) / downside_std * np.sqrt(252)
        except Exception as e:
            self.logger.error(f"소르티노 비율 계산 중 오류 발생: {e}")
            return None
            
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """칼마 비율 계산"""
        try:
            annual_return = self._calculate_annual_return(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            return annual_return / max_drawdown
        except Exception as e:
            self.logger.error(f"칼마 비율 계산 중 오류 발생: {e}")
            return None
            
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """승률 계산"""
        try:
            winning_days = np.sum(returns > 0)
            total_days = len(returns)
            return winning_days / total_days
        except Exception as e:
            self.logger.error(f"승률 계산 중 오류 발생: {e}")
            return None
            
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """수익 인자 계산"""
        try:
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return np.inf
            return np.sum(positive_returns) / abs(np.sum(negative_returns))
        except Exception as e:
            self.logger.error(f"수익 인자 계산 중 오류 발생: {e}")
            return None
            
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """최대 손실폭 계산"""
        try:
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (running_max - cumulative_returns) / running_max
            return np.max(drawdown)
        except Exception as e:
            self.logger.error(f"최대 손실폭 계산 중 오류 발생: {e}")
            return None
            
    def _calculate_recovery_factor(self, returns: np.ndarray) -> float:
        """회복 인자 계산"""
        try:
            total_return = self._calculate_total_return(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            return total_return / max_drawdown
        except Exception as e:
            self.logger.error(f"회복 인자 계산 중 오류 발생: {e}")
            return None
            
    def analyze_risk_contribution(self) -> Dict[str, float]:
        """
        리스크 기여도 분석
        
        Returns:
            리스크 기여도 분석 결과 딕셔너리
        """
        try:
            portfolio_volatility = np.sqrt(np.dot(self.weights.T, np.dot(self.returns.cov(), self.weights)))
            marginal_contribution = np.dot(self.returns.cov(), self.weights) / portfolio_volatility
            risk_contribution = self.weights * marginal_contribution
            
            analysis = {
                "total_risk": portfolio_volatility,
                "risk_contributions": dict(zip(self.returns.columns, risk_contribution)),
                "risk_contribution_ratios": dict(zip(self.returns.columns, risk_contribution / portfolio_volatility))
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"리스크 기여도 분석 중 오류 발생: {e}")
            return {}
            
    def analyze_correlation(self) -> Dict[str, float]:
        """
        상관관계 분석
        
        Returns:
            상관관계 분석 결과 딕셔너리
        """
        try:
            correlation_matrix = self.returns.corr()
            portfolio_correlation = np.dot(np.dot(self.weights, correlation_matrix), self.weights)
            
            analysis = {
                "correlation_matrix": correlation_matrix.to_dict(),
                "portfolio_correlation": portfolio_correlation,
                "average_correlation": np.mean(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)])
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"상관관계 분석 중 오류 발생: {e}")
            return {} 