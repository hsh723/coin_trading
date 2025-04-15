import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import json
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class RiskManager:
    """
    리스크 관리 시스템
    
    주요 기능:
    - 포지션 사이징
    - 리스크 한도 관리
    - VaR 계산
    - 스트레스 테스트
    - 상관관계 분석
    """
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 max_position_size: float = 0.1,
                 max_drawdown: float = 0.2,
                 var_confidence: float = 0.95,
                 save_dir: str = "./risk_management"):
        """
        리스크 관리 시스템 초기화
        
        Args:
            initial_capital: 초기 자본금
            max_position_size: 최대 포지션 크기 (자본금 대비)
            max_drawdown: 최대 허용 낙폭
            var_confidence: VaR 신뢰 수준
            save_dir: 결과 저장 디렉토리
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.var_confidence = var_confidence
        self.save_dir = save_dir
        
        # 결과 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 리스크 메트릭 저장용 변수
        self.positions = []
        self.portfolio_values = []
        self.drawdowns = []
        self.var_values = []
        self.correlations = []
    
    def calculate_position_size(self,
                              price: float,
                              volatility: float,
                              correlation: float = 0.0) -> float:
        """
        포지션 사이즈 계산
        
        Args:
            price: 현재 가격
            volatility: 변동성
            correlation: 상관관계
            
        Returns:
            포지션 사이즈
        """
        # 켈리 크라이테리온 기반 포지션 사이징
        win_rate = 0.5  # 기본 승률
        win_loss_ratio = 2.0  # 기본 승/패 비율
        
        # 켈리 공식 적용
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # 변동성 조정
        volatility_adjustment = 1 / (1 + volatility)
        
        # 상관관계 조정
        correlation_adjustment = 1 / (1 + abs(correlation))
        
        # 최종 포지션 사이즈 계산
        position_size = (self.initial_capital * kelly_fraction * 
                        volatility_adjustment * correlation_adjustment)
        
        # 최대 포지션 사이즈 제한
        position_size = min(position_size, self.initial_capital * self.max_position_size)
        
        return position_size
    
    def calculate_var(self,
                     returns: pd.Series,
                     confidence_level: float = None) -> float:
        """
        VaR (Value at Risk) 계산
        
        Args:
            returns: 수익률 시계열
            confidence_level: 신뢰 수준
            
        Returns:
            VaR 값
        """
        confidence_level = confidence_level or self.var_confidence
        
        # 히스토리컬 VaR 계산
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        # 파라미트릭 VaR 계산 (정규분포 가정)
        mean = returns.mean()
        std = returns.std()
        parametric_var = stats.norm.ppf(1 - confidence_level, mean, std)
        
        # 두 방법의 평균 사용
        final_var = (var + parametric_var) / 2
        
        return final_var
    
    def perform_stress_test(self,
                          data: pd.DataFrame,
                          scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        스트레스 테스트 수행
        
        Args:
            data: 시계열 데이터
            scenarios: 스트레스 시나리오
            
        Returns:
            스트레스 테스트 결과
        """
        logger.info("스트레스 테스트 시작...")
        
        results = {}
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', 'unnamed_scenario')
            price_shock = scenario.get('price_shock', 0.0)
            volatility_shock = scenario.get('volatility_shock', 0.0)
            
            # 시나리오 적용
            shocked_data = data.copy()
            shocked_data['price'] *= (1 + price_shock)
            
            # 변동성 계산
            returns = shocked_data['price'].pct_change().dropna()
            volatility = returns.std() * (1 + volatility_shock)
            
            # VaR 계산
            var = self.calculate_var(returns)
            
            # 최대 낙폭 계산
            portfolio_values = shocked_data['price'].cumprod()
            max_drawdown = (portfolio_values / portfolio_values.cummax() - 1).min()
            
            # 결과 저장
            results[scenario_name] = {
                'price_shock': price_shock,
                'volatility_shock': volatility_shock,
                'var': var,
                'max_drawdown': max_drawdown,
                'volatility': volatility
            }
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.save_dir, f"stress_test_results_{timestamp}.json")
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        return results
    
    def analyze_correlations(self,
                           data: pd.DataFrame,
                           window: int = 30) -> Dict[str, Any]:
        """
        상관관계 분석
        
        Args:
            data: 시계열 데이터
            window: 이동 윈도우 크기
            
        Returns:
            상관관계 분석 결과
        """
        logger.info("상관관계 분석 시작...")
        
        # 수익률 계산
        returns = data.pct_change().dropna()
        
        # 이동 상관관계 계산
        rolling_corr = returns.rolling(window=window).corr()
        
        # 상관관계 통계
        mean_corr = rolling_corr.mean()
        std_corr = rolling_corr.std()
        max_corr = rolling_corr.max()
        min_corr = rolling_corr.min()
        
        # 결과 저장
        correlation_results = {
            'mean_correlation': mean_corr,
            'std_correlation': std_corr,
            'max_correlation': max_corr,
            'min_correlation': min_corr,
            'rolling_correlations': rolling_corr
        }
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.save_dir, f"correlation_results_{timestamp}.json")
        
        with open(result_file, 'w') as f:
            json.dump(correlation_results, f, indent=4, default=str)
        
        return correlation_results
    
    def monitor_risk_metrics(self,
                           portfolio_value: float,
                           position: float,
                           price: float) -> Dict[str, Any]:
        """
        리스크 메트릭 모니터링
        
        Args:
            portfolio_value: 포트폴리오 가치
            position: 현재 포지션
            price: 현재 가격
            
        Returns:
            리스크 메트릭
        """
        # 낙폭 계산
        self.portfolio_values.append(portfolio_value)
        max_portfolio_value = max(self.portfolio_values)
        drawdown = (portfolio_value - max_portfolio_value) / max_portfolio_value
        
        # 포지션 크기 계산
        position_size = abs(position * price) / self.initial_capital
        
        # 리스크 메트릭 저장
        risk_metrics = {
            'drawdown': drawdown,
            'position_size': position_size,
            'portfolio_value': portfolio_value,
            'max_drawdown': min(self.drawdowns + [drawdown]) if self.drawdowns else drawdown
        }
        
        # 경고 체크
        warnings = []
        if drawdown < -self.max_drawdown:
            warnings.append("최대 낙폭 초과")
        if position_size > self.max_position_size:
            warnings.append("최대 포지션 크기 초과")
        
        risk_metrics['warnings'] = warnings
        
        return risk_metrics 