import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import logging

from .optimizer import PortfolioOptimizer

class DynamicAllocationManager:
    """동적 자산 배분 및 재조정 관리 클래스
    
    다양한 포트폴리오 재조정 전략 지원:
    - 기간 기반 재조정 (정기적)
    - 임계값 기반 재조정 (가중치 변동 시)
    - 복합 재조정 전략
    - 시장 상황 기반 재조정
    """
    
    def __init__(self, initial_weights: Dict[str, float], 
                config: Dict[str, Any] = None):
        """
        Args:
            initial_weights: 초기 자산 가중치
            config: 설정 파라미터
        """
        self.target_weights = initial_weights
        self.current_weights = initial_weights.copy()
        
        # 기본 설정
        self.config = {
            'rebalance_threshold': 0.05,  # 재조정 임계값 (5%)
            'rebalance_period': 30,       # 재조정 주기 (일)
            'min_rebalance_interval': 7,  # 최소 재조정 간격 (일)
            'max_single_trade': 0.2,      # 최대 단일 거래 크기 (20%)
            'trading_cost': 0.002,        # 거래 비용 (0.2%)
            'use_tactical_shifts': False,  # 전술적 조정 사용
            'tactical_range': 0.1,        # 전술적 조정 범위 (±10%)
            'risk_based_thresholds': False, # 리스크 기반 임계값
            'auto_update_target': False,   # 목표 가중치 자동 업데이트
            'update_frequency': 90        # 목표 가중치 업데이트 주기 (일)
        }
        
        # 사용자 설정으로 기본 설정 업데이트
        if config:
            self.config.update(config)
        
        self.last_rebalance_date = None
        self.last_target_update = None
        self.rebalance_history = []
        self.logger = logging.getLogger(__name__)
    
    def get_current_weights(self) -> Dict[str, float]:
        """현재 자산 가중치 반환"""
        return self.current_weights
    
    def get_target_weights(self) -> Dict[str, float]:
        """목표 자산 가중치 반환"""
        return self.target_weights
    
    def update_current_weights(self, weights: Dict[str, float]) -> None:
        """현재 자산 가중치 업데이트
        
        Args:
            weights: 새로운 자산 가중치
        """
        # 모든 자산 포함 확인
        missing_assets = set(self.current_weights.keys()) - set(weights.keys())
        if missing_assets:
            raise ValueError(f"누락된 자산이 있습니다: {missing_assets}")
        
        self.current_weights = weights.copy()
    
    def set_target_weights(self, weights: Dict[str, float]) -> None:
        """목표 자산 가중치 설정
        
        Args:
            weights: 새로운 목표 가중치
        """
        # 가중치 합계 확인
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            self.logger.warning(f"가중치 합계가 1이 아닙니다: {total_weight}. 정규화합니다.")
            weights = {asset: w/total_weight for asset, w in weights.items()}
        
        self.target_weights = weights.copy()
        self.last_target_update = datetime.now()
    
    def should_rebalance(self, current_date: datetime = None) -> Tuple[bool, str]:
        """재조정 필요 여부 확인
        
        Args:
            current_date: 현재 날짜 (None이면 현재 시간 사용)
        
        Returns:
            (재조정 필요 여부, 사유)
        """
        if current_date is None:
            current_date = datetime.now()
        
        # 최소 재조정 간격 확인
        if self.last_rebalance_date is not None:
            days_since_last = (current_date - self.last_rebalance_date).days
            min_interval = self.config.get('min_rebalance_interval', 7)
            
            if days_since_last < min_interval:
                return False, f"최소 재조정 간격 미달 ({days_since_last}/{min_interval}일)"
        
        # 1. 기간 기반 재조정
        rebalance_period = self.config.get('rebalance_period', 30)
        if self.last_rebalance_date is None:
            period_trigger = True
            reason = "최초 재조정"
        else:
            days_since_last = (current_date - self.last_rebalance_date).days
            period_trigger = days_since_last >= rebalance_period
            reason = f"정기 재조정 기간 도달 ({days_since_last}/{rebalance_period}일)"
        
        if period_trigger:
            return True, reason
        
        # 2. 임계값 기반 재조정
        threshold = self.config.get('rebalance_threshold', 0.05)
        
        # 리스크 기반 임계값 사용 시 자산별 임계값 계산
        if self.config.get('risk_based_thresholds', False):
            return self._check_risk_based_threshold(threshold)
        
        # 일반 임계값 사용
        for asset in self.target_weights:
            target = self.target_weights.get(asset, 0)
            current = self.current_weights.get(asset, 0)
            
            # 절대 편차 계산
            deviation = abs(current - target)
            
            # 임계값 초과 시 재조정 필요
            if deviation > threshold:
                return True, f"자산 {asset}의 가중치 편차 초과 ({deviation:.3f} > {threshold:.3f})"
        
        return False, "재조정 불필요"
    
    def _check_risk_based_threshold(self, base_threshold: float) -> Tuple[bool, str]:
        """리스크 기반 임계값 확인
        
        중요도가 높은 자산(가중치가 큰 자산)은 더 작은 임계값 적용
        
        Args:
            base_threshold: 기본 임계값
            
        Returns:
            (재조정 필요 여부, 사유)
        """
        for asset in self.target_weights:
            target = self.target_weights.get(asset, 0)
            current = self.current_weights.get(asset, 0)
            
            # 자산별 임계값 계산: 가중치가 높을수록 더 작은 임계값
            asset_threshold = base_threshold * (1 - 0.5 * target)
            
            # 절대 편차 계산
            deviation = abs(current - target)
            
            # 임계값 초과 시 재조정 필요
            if deviation > asset_threshold:
                return True, f"자산 {asset}의 리스크 기반 임계값 초과 ({deviation:.3f} > {asset_threshold:.3f})"
        
        return False, "리스크 기반 재조정 불필요"
    
    def calculate_rebalance_trades(self, current_value: float) -> Dict[str, float]:
        """재조정을 위한 거래 계산
        
        Args:
            current_value: 현재 포트폴리오 총 가치
            
        Returns:
            자산별 거래 금액 (양수: 매수, 음수: 매도)
        """
        trades = {}
        
        # 전술적 조정 적용 (설정된 경우)
        adjusted_targets = self._apply_tactical_adjustments()
        
        # 자산별 거래 금액 계산
        for asset in set(self.current_weights.keys()) | set(adjusted_targets.keys()):
            target_weight = adjusted_targets.get(asset, 0)
            current_weight = self.current_weights.get(asset, 0)
            
            target_value = current_value * target_weight
            current_value_asset = current_value * current_weight
            
            # 거래 금액
            trade_value = target_value - current_value_asset
            
            # 최대 단일 거래 크기 제한
            max_trade = current_value * self.config.get('max_single_trade', 0.2)
            if abs(trade_value) > max_trade:
                trade_value = np.sign(trade_value) * max_trade
            
            # 최소 거래 금액보다 큰 경우만 거래 실행
            min_trade_value = current_value * 0.001  # 0.1% 이상
            if abs(trade_value) > min_trade_value:
                trades[asset] = trade_value
        
        return trades
    
    def _apply_tactical_adjustments(self) -> Dict[str, float]:
        """전술적 자산 배분 조정 적용
        
        시장 상황에 따라 목표 가중치를 전술적으로 조정
        
        Returns:
            전술적으로 조정된 목표 가중치
        """
        # 전술적 조정을 사용하지 않으면 원래 목표 반환
        if not self.config.get('use_tactical_shifts', False):
            return self.target_weights.copy()
        
        adjusted_weights = self.target_weights.copy()
        tactical_range = self.config.get('tactical_range', 0.1)
        
        # 여기에 전술적 조정 로직 구현
        # 예: 시장 변동성, 모멘텀, 가치평가 등에 기반한 조정
        # 지금은 간단한 예시만 구현
        
        # 정규화하여 합이 1이 되도록 함
        total = sum(adjusted_weights.values())
        return {asset: w/total for asset, w in adjusted_weights.items()}
    
    def execute_rebalance(self, current_value: float, 
                         current_date: datetime = None) -> Dict[str, Any]:
        """포트폴리오 재조정 실행
        
        Args:
            current_value: 현재 포트폴리오 총 가치
            current_date: 현재 날짜 (None이면 현재 시간 사용)
            
        Returns:
            재조정 결과 정보
        """
        if current_date is None:
            current_date = datetime.now()
        
        # 재조정 필요 여부 확인
        should_rebal, reason = self.should_rebalance(current_date)
        
        if not should_rebal:
            return {
                'rebalanced': False,
                'reason': reason,
                'trades': {},
                'old_weights': self.current_weights.copy(),
                'new_weights': self.current_weights.copy(),
                'cost': 0.0
            }
        
        # 거래 계산
        trades = self.calculate_rebalance_trades(current_value)
        
        # 거래 비용 계산
        trading_cost = self.config.get('trading_cost', 0.002)
        cost = sum(abs(trade) for trade in trades.values()) * trading_cost
        
        # 새 가중치 계산
        new_weights = self.current_weights.copy()
        for asset, trade_value in trades.items():
            # 포트폴리오에 새 자산 추가
            if asset not in new_weights:
                new_weights[asset] = 0
            
            # 가중치 업데이트
            new_weights[asset] += trade_value / current_value
        
        # 정규화
        total_weight = sum(new_weights.values())
        new_weights = {asset: w/total_weight for asset, w in new_weights.items()}
        
        # 매우 작은 가중치 제거 (0.1% 미만)
        new_weights = {asset: w for asset, w in new_weights.items() if w >= 0.001}
        
        # 정규화
        total_weight = sum(new_weights.values())
        new_weights = {asset: w/total_weight for asset, w in new_weights.items()}
        
        # 현재 가중치 업데이트
        self.current_weights = new_weights
        
        # 재조정 기록 저장
        rebalance_record = {
            'date': current_date,
            'reason': reason,
            'old_weights': self.current_weights.copy(),
            'new_weights': new_weights,
            'trades': trades,
            'cost': cost,
            'portfolio_value': current_value
        }
        
        self.rebalance_history.append(rebalance_record)
        self.last_rebalance_date = current_date
        
        # 목표 가중치 자동 업데이트 (설정된 경우)
        if self.config.get('auto_update_target', False):
            self._check_target_update(current_date)
        
        return {
            'rebalanced': True,
            'reason': reason,
            'trades': trades,
            'old_weights': self.current_weights.copy(),
            'new_weights': new_weights,
            'cost': cost
        }
    
    def _check_target_update(self, current_date: datetime) -> None:
        """목표 가중치 자동 업데이트 확인
        
        Args:
            current_date: 현재 날짜
        """
        if self.last_target_update is None:
            return
        
        update_frequency = self.config.get('update_frequency', 90)
        days_since_update = (current_date - self.last_target_update).days
        
        if days_since_update >= update_frequency:
            self.logger.info(f"목표 가중치 자동 업데이트 시간 ({days_since_update}일 경과)")
            # 업데이트 로직은 구현하지 않음 - 외부에서 set_target_weights 호출 필요
    
    def get_rebalance_history(self) -> List[Dict[str, Any]]:
        """재조정 히스토리 반환"""
        return self.rebalance_history
    
    def simulate_rebalance_strategy(self, price_history: pd.DataFrame, 
                                 initial_value: float = 10000,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> pd.DataFrame:
        """재조정 전략 시뮬레이션
        
        Args:
            price_history: 가격 이력 데이터프레임 (각 열은 자산)
            initial_value: 초기 포트폴리오 가치
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            시뮬레이션 결과 데이터프레임
        """
        # 날짜 필터링
        if start_date is not None:
            price_history = price_history[price_history.index >= start_date]
        if end_date is not None:
            price_history = price_history[price_history.index <= end_date]
        
        # 시뮬레이션 결과 저장용 데이터프레임
        results = pd.DataFrame(index=price_history.index)
        results['portfolio_value'] = np.nan
        
        # 포트폴리오 초기화
        self.last_rebalance_date = None
        self.rebalance_history = []
        
        # 초기 자산 수량 계산
        current_weights = self.target_weights.copy()
        self.current_weights = current_weights
        
        current_value = initial_value
        asset_quantities = {}
        
        # 초기 날짜의 가격으로 자산 수량 계산
        initial_prices = price_history.iloc[0]
        for asset, weight in current_weights.items():
            if asset in initial_prices:
                quantity = (weight * current_value) / initial_prices[asset]
                asset_quantities[asset] = quantity
            else:
                self.logger.warning(f"가격 데이터에 자산 {asset}이 없습니다")
        
        # 각 날짜에 대해 시뮬레이션
        for date, prices in price_history.iterrows():
            # 현재 포트폴리오 가치 계산
            current_value = 0
            for asset, quantity in asset_quantities.items():
                if asset in prices:
                    current_value += quantity * prices[asset]
            
            # 현재 가중치 계산
            current_weights = {}
            for asset, quantity in asset_quantities.items():
                if asset in prices:
                    current_weights[asset] = (quantity * prices[asset]) / current_value
            
            self.current_weights = current_weights
            
            # 재조정 실행
            rebalance_result = self.execute_rebalance(current_value, date)
            
            # 재조정이 실행되었다면 자산 수량 조정
            if rebalance_result['rebalanced']:
                for asset, trade_value in rebalance_result['trades'].items():
                    if asset in prices:
                        # 거래에 따른 수량 변화
                        quantity_change = trade_value / prices[asset]
                        
                        # 자산이 없으면 추가
                        if asset not in asset_quantities:
                            asset_quantities[asset] = 0
                        
                        # 수량 업데이트
                        asset_quantities[asset] += quantity_change
                
                # 거래 비용 차감
                current_value -= rebalance_result['cost']
            
            # 결과 저장
            results.loc[date, 'portfolio_value'] = current_value
            for asset in asset_quantities:
                if asset in prices:
                    results.loc[date, f'weight_{asset}'] = (asset_quantities[asset] * prices[asset]) / current_value
                    results.loc[date, f'quantity_{asset}'] = asset_quantities[asset]
            
            results.loc[date, 'rebalanced'] = int(rebalance_result['rebalanced'])
            if rebalance_result['rebalanced']:
                results.loc[date, 'rebalance_reason'] = rebalance_result['reason']
                results.loc[date, 'rebalance_cost'] = rebalance_result['cost']
        
        # 수익률 계산
        results['return'] = results['portfolio_value'].pct_change()
        results['cumulative_return'] = (1 + results['return']).cumprod() - 1
        
        return results 