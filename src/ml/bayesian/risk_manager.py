import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import os
import json
from scipy import stats
from scipy.stats import norm

logger = logging.getLogger(__name__)

class RiskManager:
    """
    리스크 관리 시스템
    
    주요 기능:
    - 포지션 사이징
    - VaR 계산
    - 스트레스 테스트
    - 상관관계 분석
    - 리스크 모니터링
    """
    
    def __init__(self,
                 config_path: str = "./config/risk_config.json",
                 data_dir: str = "./data",
                 log_dir: str = "./logs"):
        """
        리스크 관리 시스템 초기화
        
        Args:
            config_path: 설정 파일 경로
            data_dir: 데이터 디렉토리
            log_dir: 로그 디렉토리
        """
        self.config_path = config_path
        self.data_dir = data_dir
        self.log_dir = log_dir
        
        # 로거 설정
        self.logger = logging.getLogger("risk_manager")
        
        # 리스크 데이터
        self.returns = None
        self.weights = None
        self.positions = None
        self.portfolio_value = None
        
        # 설정 로드
        self.config = self._load_config()
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)
        self.max_drawdown = self.config.get("max_drawdown", 0.2)
        self.position_limit = self.config.get("position_limit", 0.2)
        self.var_confidence = self.config.get("var_confidence", 0.95)
        self.volatility_threshold = self.config.get("volatility_threshold", 0.3)
        
        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 포지션 정보
        self.positions = {}
        
        # 메트릭스
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'var': 0.0
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
            
    def update_data(self,
                   returns: pd.DataFrame,
                   weights: np.ndarray,
                   positions: Dict[str, float],
                   portfolio_value: float) -> None:
        """
        리스크 데이터 업데이트
        
        Args:
            returns: 자산별 수익률 데이터프레임
            weights: 자산별 가중치 배열
            positions: 자산별 포지션 크기
            portfolio_value: 포트폴리오 가치
        """
        try:
            self.returns = returns
            self.weights = weights
            self.positions = positions
            self.portfolio_value = portfolio_value
            self.logger.info("리스크 데이터 업데이트 완료")
        except Exception as e:
            self.logger.error(f"데이터 업데이트 중 오류 발생: {e}")
            
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
        try:
            # 켈리 크라이테리온 기반 포지션 사이징
            win_rate = self.metrics['winning_trades'] / max(1, self.metrics['total_trades'])
            win_loss_ratio = 2.0  # 기본값
            
            if win_rate > 0:
                kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
            else:
                kelly_fraction = 0.1  # 기본값
            
            # 변동성 조정
            volatility_adjustment = 1 / (1 + volatility)
            
            # 상관관계 조정
            correlation_adjustment = 1 / (1 + abs(correlation))
            
            # 최종 포지션 사이즈 계산
            position_size = (self.portfolio_value * kelly_fraction * 
                           volatility_adjustment * correlation_adjustment)
            
            # 최대 포지션 사이즈 제한
            max_allowed = self.portfolio_value * self.position_limit
            position_size = min(position_size, max_allowed)
            
            return position_size
            
        except Exception as e:
            logger.error(f"포지션 사이즈 계산 중 오류 발생: {e}")
            return 0.0
    
    def calculate_var(self, returns: pd.Series, window: int = 252) -> float:
        """
        VaR 계산
        
        Args:
            returns: 수익률 시리즈
            window: 윈도우 크기
            
        Returns:
            VaR 값
        """
        try:
            # 수익률 정규화
            returns = returns.dropna()
            
            if len(returns) < window:
                return 0.0
            
            # 과거 VaR
            historical_var = np.percentile(returns, (1 - self.var_confidence) * 100)
            
            # 파라메트릭 VaR
            mu = returns.mean()
            sigma = returns.std()
            parametric_var = stats.norm.ppf(1 - self.var_confidence, mu, sigma)
            
            # 두 방법의 평균
            var = (historical_var + parametric_var) / 2
            
            # 메트릭스 업데이트
            self.metrics['var'] = var
            
            return var
            
        except Exception as e:
            logger.error(f"VaR 계산 중 오류 발생: {e}")
            return 0.0
    
    def perform_stress_test(self,
                          scenarios: Dict[str, Dict[str, float]],
                          save_report: bool = True) -> Dict[str, Any]:
        """
        스트레스 테스트 수행
        
        Args:
            scenarios: 시나리오 딕셔너리
            save_report: 보고서 저장 여부
            
        Returns:
            스트레스 테스트 결과
        """
        try:
            results = {}
            
            for scenario_name, scenario in scenarios.items():
                # 초기 자본금으로 리셋
                capital = self.portfolio_value
                
                # 시나리오 적용
                for symbol, change in scenario.items():
                    if symbol in self.positions:
                        position = self.positions[symbol]
                        capital += position * self.weights[symbol] * change
                
                # 결과 저장
                results[scenario_name] = {
                    'final_capital': capital,
                    'return': (capital - self.portfolio_value) / self.portfolio_value
                }
            
            # 보고서 저장
            if save_report:
                filename = f"{self.log_dir}/stress_test_{datetime.now().strftime('%Y%m%d')}.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)
                logger.info(f"스트레스 테스트 보고서 저장 완료: {filename}")
            
            return results
            
        except Exception as e:
            logger.error(f"스트레스 테스트 중 오류 발생: {e}")
            return {}
    
    def analyze_correlations(self,
                           returns: Dict[str, pd.Series],
                           window: int = 20) -> pd.DataFrame:
        """
        상관관계 분석
        
        Args:
            returns: 수익률 딕셔너리
            window: 롤링 윈도우 크기
            
        Returns:
            상관관계 데이터프레임
        """
        try:
            # 수익률 데이터프레임 생성
            df = pd.DataFrame(returns)
            
            # 롤링 상관관계 계산
            correlations = df.rolling(window=window).corr()
            
            # 결과 저장
            filename = f"{self.log_dir}/correlations_{datetime.now().strftime('%Y%m%d')}.csv"
            correlations.to_csv(filename)
            logger.info(f"상관관계 분석 결과 저장 완료: {filename}")
            
            return correlations
            
        except Exception as e:
            logger.error(f"상관관계 분석 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def monitor_risk(self) -> Dict[str, Any]:
        """
        리스크 모니터링
        
        Returns:
            리스크 모니터링 결과 딕셔너리
        """
        try:
            monitoring = {
                "position_risk": self._check_position_risk(),
                "drawdown_risk": self._check_drawdown_risk(),
                "volatility_risk": self._check_volatility_risk(),
                "var_risk": self._check_var_risk(),
                "correlation_risk": self._check_correlation_risk(),
                "liquidity_risk": self._check_liquidity_risk()
            }
            
            return monitoring
            
        except Exception as e:
            logger.error(f"리스크 모니터링 중 오류 발생: {e}")
            return {}
            
    def _check_position_risk(self) -> Dict[str, Any]:
        """포지션 리스크 확인"""
        try:
            position_risk = {}
            for asset, position in self.positions.items():
                position_ratio = abs(position) / self.portfolio_value
                position_risk[asset] = {
                    "position_ratio": position_ratio,
                    "exceeds_limit": position_ratio > self.position_limit
                }
            return position_risk
        except Exception as e:
            logger.error(f"포지션 리스크 확인 중 오류 발생: {e}")
            return {}
            
    def _check_drawdown_risk(self) -> Dict[str, Any]:
        """드로다운 리스크 확인"""
        try:
            portfolio_returns = np.dot(self.returns, self.weights)
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (running_max - cumulative_returns) / running_max
            current_drawdown = drawdown[-1]
            
            return {
                "current_drawdown": current_drawdown,
                "exceeds_limit": current_drawdown > self.max_drawdown,
                "max_drawdown": np.max(drawdown)
            }
        except Exception as e:
            logger.error(f"드로다운 리스크 확인 중 오류 발생: {e}")
            return {}
            
    def _check_volatility_risk(self) -> Dict[str, Any]:
        """변동성 리스크 확인"""
        try:
            portfolio_returns = np.dot(self.returns, self.weights)
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            
            return {
                "volatility": volatility,
                "exceeds_threshold": volatility > self.volatility_threshold
            }
        except Exception as e:
            logger.error(f"변동성 리스크 확인 중 오류 발생: {e}")
            return {}
            
    def _check_var_risk(self) -> Dict[str, Any]:
        """VaR 리스크 확인"""
        try:
            portfolio_returns = np.dot(self.returns, self.weights)
            var = np.percentile(portfolio_returns, (1 - self.var_confidence) * 100)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            
            return {
                "var": var,
                "cvar": cvar,
                "exceeds_threshold": var < -0.05  # 임계값 설정
            }
        except Exception as e:
            logger.error(f"VaR 리스크 확인 중 오류 발생: {e}")
            return {}
            
    def _check_correlation_risk(self) -> Dict[str, Any]:
        """상관관계 리스크 확인"""
        try:
            correlation_matrix = self.returns.corr()
            portfolio_correlation = np.dot(np.dot(self.weights, correlation_matrix), self.weights)
            
            return {
                "portfolio_correlation": portfolio_correlation,
                "high_correlation": portfolio_correlation > 0.8  # 임계값 설정
            }
        except Exception as e:
            logger.error(f"상관관계 리스크 확인 중 오류 발생: {e}")
            return {}
            
    def _check_liquidity_risk(self) -> Dict[str, Any]:
        """유동성 리스크 확인"""
        try:
            # 거래량 기반 유동성 지표 계산
            volume_risk = {}
            for asset in self.positions.keys():
                if asset in self.returns.columns:
                    volume = self.returns[asset].count() / len(self.returns)
                    position_value = abs(self.positions[asset])
                    volume_risk[asset] = {
                        "volume_ratio": volume,
                        "position_ratio": position_value / self.portfolio_value,
                        "liquidity_risk": position_value / (volume * self.portfolio_value)
                    }
            return volume_risk
        except Exception as e:
            logger.error(f"유동성 리스크 확인 중 오류 발생: {e}")
            return {}
            
    def adjust_positions(self) -> Dict[str, float]:
        """
        포지션 조정
        
        Returns:
            조정된 포지션 딕셔너리
        """
        try:
            risk_monitoring = self.monitor_risk()
            adjusted_positions = self.positions.copy()
            
            # 포지션 리스크 조정
            position_risk = risk_monitoring.get("position_risk", {})
            for asset, risk in position_risk.items():
                if risk["exceeds_limit"]:
                    adjusted_positions[asset] = self.positions[asset] * self.position_limit / risk["position_ratio"]
                    
            # 드로다운 리스크 조정
            drawdown_risk = risk_monitoring.get("drawdown_risk", {})
            if drawdown_risk.get("exceeds_limit", False):
                reduction_factor = 1 - (drawdown_risk["current_drawdown"] / self.max_drawdown)
                for asset in adjusted_positions:
                    adjusted_positions[asset] *= reduction_factor
                    
            # 변동성 리스크 조정
            volatility_risk = risk_monitoring.get("volatility_risk", {})
            if volatility_risk.get("exceeds_threshold", False):
                reduction_factor = self.volatility_threshold / volatility_risk["volatility"]
                for asset in adjusted_positions:
                    adjusted_positions[asset] *= reduction_factor
                    
            return adjusted_positions
            
        except Exception as e:
            logger.error(f"포지션 조정 중 오류 발생: {e}")
            return self.positions
            
    def generate_risk_report(self) -> Dict[str, Any]:
        """
        리스크 보고서 생성
        
        Returns:
            리스크 보고서 딕셔너리
        """
        try:
            risk_monitoring = self.monitor_risk()
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": self.portfolio_value,
                "risk_metrics": {
                    "position_risk": risk_monitoring.get("position_risk", {}),
                    "drawdown_risk": risk_monitoring.get("drawdown_risk", {}),
                    "volatility_risk": risk_monitoring.get("volatility_risk", {}),
                    "var_risk": risk_monitoring.get("var_risk", {}),
                    "correlation_risk": risk_monitoring.get("correlation_risk", {}),
                    "liquidity_risk": risk_monitoring.get("liquidity_risk", {})
                },
                "risk_warnings": self._generate_risk_warnings(risk_monitoring),
                "recommendations": self._generate_recommendations(risk_monitoring)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"리스크 보고서 생성 중 오류 발생: {e}")
            return {}
            
    def _generate_risk_warnings(self, risk_monitoring: Dict[str, Any]) -> List[str]:
        """리스크 경고 생성"""
        try:
            warnings = []
            
            # 포지션 리스크 경고
            position_risk = risk_monitoring.get("position_risk", {})
            for asset, risk in position_risk.items():
                if risk["exceeds_limit"]:
                    warnings.append(f"포지션 리스크: {asset}의 포지션 비율이 한도를 초과했습니다.")
                    
            # 드로다운 리스크 경고
            drawdown_risk = risk_monitoring.get("drawdown_risk", {})
            if drawdown_risk.get("exceeds_limit", False):
                warnings.append(f"드로다운 리스크: 현재 드로다운이 한도를 초과했습니다.")
                
            # 변동성 리스크 경고
            volatility_risk = risk_monitoring.get("volatility_risk", {})
            if volatility_risk.get("exceeds_threshold", False):
                warnings.append(f"변동성 리스크: 포트폴리오 변동성이 임계값을 초과했습니다.")
                
            # VaR 리스크 경고
            var_risk = risk_monitoring.get("var_risk", {})
            if var_risk.get("exceeds_threshold", False):
                warnings.append(f"VaR 리스크: VaR이 임계값을 초과했습니다.")
                
            # 상관관계 리스크 경고
            correlation_risk = risk_monitoring.get("correlation_risk", {})
            if correlation_risk.get("high_correlation", False):
                warnings.append(f"상관관계 리스크: 포트폴리오 상관관계가 높습니다.")
                
            return warnings
            
        except Exception as e:
            logger.error(f"리스크 경고 생성 중 오류 발생: {e}")
            return []
            
    def _generate_recommendations(self, risk_monitoring: Dict[str, Any]) -> List[str]:
        """리스크 관리 권장사항 생성"""
        try:
            recommendations = []
            
            # 포지션 조정 권장
            position_risk = risk_monitoring.get("position_risk", {})
            for asset, risk in position_risk.items():
                if risk["exceeds_limit"]:
                    recommendations.append(f"{asset}의 포지션을 {self.position_limit * 100}%로 축소하세요.")
                    
            # 드로다운 관리 권장
            drawdown_risk = risk_monitoring.get("drawdown_risk", {})
            if drawdown_risk.get("exceeds_limit", False):
                recommendations.append("드로다운을 관리하기 위해 포지션을 축소하세요.")
                
            # 변동성 관리 권장
            volatility_risk = risk_monitoring.get("volatility_risk", {})
            if volatility_risk.get("exceeds_threshold", False):
                recommendations.append("변동성을 줄이기 위해 포지션을 조정하세요.")
                
            # 리스크 분산 권장
            correlation_risk = risk_monitoring.get("correlation_risk", {})
            if correlation_risk.get("high_correlation", False):
                recommendations.append("리스크를 분산하기 위해 상관관계가 낮은 자산을 추가하세요.")
                
            return recommendations
            
        except Exception as e:
            logger.error(f"리스크 관리 권장사항 생성 중 오류 발생: {e}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """리스크 메트릭스 반환"""
        return self.metrics 