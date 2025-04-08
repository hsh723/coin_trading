"""
A/B 테스트 프레임워크 모듈
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

class ABTestingFramework:
    """A/B 테스트 프레임워크 클래스"""
    
    def __init__(self, db_manager):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.active_tests = {}
        
    def start_test(self,
                  test_name: str,
                  control_strategy: str,
                  treatment_strategy: str,
                  test_duration: int = 7,
                  sample_size: int = 100) -> bool:
        """
        A/B 테스트 시작
        
        Args:
            test_name (str): 테스트 이름
            control_strategy (str): 대조군 전략
            treatment_strategy (str): 실험군 전략
            test_duration (int): 테스트 기간(일)
            sample_size (int): 샘플 크기
            
        Returns:
            bool: 테스트 시작 성공 여부
        """
        try:
            # 테스트 설정 저장
            test_config = {
                'control_strategy': control_strategy,
                'treatment_strategy': treatment_strategy,
                'start_time': datetime.now(),
                'end_time': datetime.now() + timedelta(days=test_duration),
                'sample_size': sample_size,
                'status': 'running'
            }
            
            self.active_tests[test_name] = test_config
            self.db_manager.save_ab_test_config(test_name, test_config)
            
            return True
            
        except Exception as e:
            self.logger.error(f"A/B 테스트 시작 실패: {str(e)}")
            return False
            
    def stop_test(self, test_name: str) -> Dict:
        """
        A/B 테스트 중지 및 결과 분석
        
        Args:
            test_name (str): 테스트 이름
            
        Returns:
            Dict: 테스트 결과
        """
        try:
            if test_name not in self.active_tests:
                raise ValueError(f"테스트 {test_name}을 찾을 수 없습니다")
                
            # 테스트 데이터 조회
            test_data = self.db_manager.get_ab_test_data(test_name)
            
            if not test_data:
                raise ValueError("테스트 데이터가 없습니다")
                
            # 결과 분석
            results = self._analyze_test_results(test_data)
            
            # 테스트 상태 업데이트
            self.active_tests[test_name]['status'] = 'completed'
            self.db_manager.update_ab_test_status(test_name, 'completed')
            
            return results
            
        except Exception as e:
            self.logger.error(f"A/B 테스트 중지 실패: {str(e)}")
            return {}
            
    def _analyze_test_results(self, test_data: List[Dict]) -> Dict:
        """
        테스트 결과 분석
        
        Args:
            test_data (List[Dict]): 테스트 데이터
            
        Returns:
            Dict: 분석 결과
        """
        try:
            # 데이터프레임 변환
            df = pd.DataFrame(test_data)
            
            # 그룹별 데이터 분리
            control_data = df[df['group'] == 'control']
            treatment_data = df[df['group'] == 'treatment']
            
            # 기본 통계 계산
            control_stats = self._calculate_group_stats(control_data)
            treatment_stats = self._calculate_group_stats(treatment_data)
            
            # 통계적 유의성 검정
            significance = self._test_significance(
                control_data['pnl'].values,
                treatment_data['pnl'].values
            )
            
            # 결과 생성
            results = {
                'control_stats': control_stats,
                'treatment_stats': treatment_stats,
                'significance': significance,
                'recommendation': self._generate_recommendation(
                    control_stats,
                    treatment_stats,
                    significance
                )
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"테스트 결과 분석 실패: {str(e)}")
            return {}
            
    def _calculate_group_stats(self, data: pd.DataFrame) -> Dict:
        """
        그룹별 통계 계산
        
        Args:
            data (pd.DataFrame): 그룹 데이터
            
        Returns:
            Dict: 통계 결과
        """
        try:
            return {
                'total_trades': len(data),
                'winning_trades': len(data[data['pnl'] > 0]),
                'losing_trades': len(data[data['pnl'] < 0]),
                'win_rate': len(data[data['pnl'] > 0]) / len(data) if len(data) > 0 else 0,
                'total_pnl': data['pnl'].sum(),
                'avg_pnl': data['pnl'].mean(),
                'std_pnl': data['pnl'].std(),
                'max_drawdown': self._calculate_max_drawdown(data['cumulative_pnl']),
                'sharpe_ratio': self._calculate_sharpe_ratio(data['pnl'])
            }
            
        except Exception as e:
            self.logger.error(f"그룹 통계 계산 실패: {str(e)}")
            return {}
            
    def _test_significance(self, control_data: np.ndarray, treatment_data: np.ndarray) -> Dict:
        """
        통계적 유의성 검정
        
        Args:
            control_data (np.ndarray): 대조군 데이터
            treatment_data (np.ndarray): 실험군 데이터
            
        Returns:
            Dict: 검정 결과
        """
        try:
            # t-검정
            t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
            
            # 효과 크기
            cohen_d = (np.mean(treatment_data) - np.mean(control_data)) / np.sqrt(
                (np.var(control_data) + np.var(treatment_data)) / 2
            )
            
            return {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': cohen_d,
                'significant': p_value < 0.05
            }
            
        except Exception as e:
            self.logger.error(f"유의성 검정 실패: {str(e)}")
            return {}
            
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """
        최대 손실폭 계산
        
        Args:
            cumulative_returns (pd.Series): 누적 수익률
            
        Returns:
            float: 최대 손실폭
        """
        try:
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            return drawdown.min()
            
        except Exception as e:
            self.logger.error(f"최대 손실폭 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        샤프 비율 계산
        
        Args:
            returns (pd.Series): 수익률
            risk_free_rate (float): 무위험 수익률
            
        Returns:
            float: 샤프 비율
        """
        try:
            excess_returns = returns - risk_free_rate / 252  # 일별 무위험 수익률
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
        except Exception as e:
            self.logger.error(f"샤프 비율 계산 실패: {str(e)}")
            return 0.0
            
    def _generate_recommendation(self,
                               control_stats: Dict,
                               treatment_stats: Dict,
                               significance: Dict) -> str:
        """
        테스트 결과에 따른 권장사항 생성
        
        Args:
            control_stats (Dict): 대조군 통계
            treatment_stats (Dict): 실험군 통계
            significance (Dict): 유의성 검정 결과
            
        Returns:
            str: 권장사항
        """
        try:
            if not significance.get('significant', False):
                return "통계적으로 유의미한 차이가 없습니다. 현재 전략을 유지하세요."
                
            if treatment_stats['total_pnl'] > control_stats['total_pnl']:
                return "실험군 전략이 더 나은 성과를 보였습니다. 실험군 전략으로 전환을 고려하세요."
            else:
                return "대조군 전략이 더 나은 성과를 보였습니다. 현재 전략을 유지하세요."
                
        except Exception as e:
            self.logger.error(f"권장사항 생성 실패: {str(e)}")
            return "결과 분석 중 오류가 발생했습니다."
            
    def get_active_tests(self) -> Dict:
        """
        진행 중인 테스트 조회
        
        Returns:
            Dict: 진행 중인 테스트 목록
        """
        return self.active_tests
        
    def get_test_status(self, test_name: str) -> Dict:
        """
        테스트 상태 조회
        
        Args:
            test_name (str): 테스트 이름
            
        Returns:
            Dict: 테스트 상태
        """
        try:
            if test_name not in self.active_tests:
                raise ValueError(f"테스트 {test_name}을 찾을 수 없습니다")
                
            return self.active_tests[test_name]
            
        except Exception as e:
            self.logger.error(f"테스트 상태 조회 실패: {str(e)}")
            return {} 