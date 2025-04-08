"""
전략 최적화 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from ..backtest.backtest_engine import BacktestEngine
from ..strategy.base_strategy import BaseStrategy
from ..utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """최적화 결과를 저장하는 데이터 클래스"""
    best_params: Dict[str, Any]
    best_score: float
    param_scores: List[Dict[str, Any]]
    optimization_history: pd.DataFrame
    performance_metrics: Dict[str, float]

class StrategyOptimizer:
    """전략 최적화 클래스"""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        param_grid: Dict[str, List[Any]],
        scoring_metric: str = 'sharpe_ratio',
        n_iter: int = 100,
        database_manager: Optional[DatabaseManager] = None
    ):
        self.strategy = strategy
        self.param_grid = param_grid
        self.scoring_metric = scoring_metric
        self.n_iter = n_iter
        self.database_manager = database_manager
        self.logger = logging.getLogger(__name__)
    
    async def optimize(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> OptimizationResult:
        """전략 최적화 실행"""
        try:
            self.logger.info("전략 최적화 시작")
            
            # 최적화 결과 저장
            optimization_history = []
            param_scores = []
            
            # 랜덤 파라미터 샘플링
            for i in range(self.n_iter):
                # 파라미터 샘플링
                params = self._sample_parameters()
                
                # 전략 파라미터 업데이트
                self.strategy.update_parameters(params)
                
                # 백테스트 실행
                engine = BacktestEngine(
                    strategy=self.strategy,
                    initial_capital=initial_capital,
                    commission=commission,
                    start_date=start_date,
                    end_date=end_date,
                    database_manager=self.database_manager
                )
                
                result = await engine.run(data)
                
                # 성과 지표 계산
                score = self._calculate_score(result)
                
                # 결과 저장
                optimization_history.append({
                    'iteration': i,
                    'params': params,
                    'score': score
                })
                
                param_scores.append({
                    'params': params,
                    'score': score,
                    'metrics': {
                        'total_return': result.total_return,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'win_rate': result.win_rate,
                        'profit_factor': result.profit_factor
                    }
                })
                
                self.logger.info(f"반복 {i+1}/{self.n_iter} 완료 - 점수: {score:.4f}")
            
            # 최적 파라미터 찾기
            best_result = max(param_scores, key=lambda x: x['score'])
            
            # 최적화 결과 생성
            result = OptimizationResult(
                best_params=best_result['params'],
                best_score=best_result['score'],
                param_scores=param_scores,
                optimization_history=pd.DataFrame(optimization_history),
                performance_metrics=best_result['metrics']
            )
            
            self.logger.info("전략 최적화 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"전략 최적화 중 오류 발생: {str(e)}")
            raise
    
    def _sample_parameters(self) -> Dict[str, Any]:
        """파라미터 샘플링"""
        params = {}
        for param, values in self.param_grid.items():
            if isinstance(values, list):
                params[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                if isinstance(values[0], int):
                    params[param] = np.random.randint(values[0], values[1])
                else:
                    params[param] = np.random.uniform(values[0], values[1])
        return params
    
    def _calculate_score(self, result: Any) -> float:
        """성과 점수 계산"""
        if self.scoring_metric == 'sharpe_ratio':
            return result.sharpe_ratio
        elif self.scoring_metric == 'total_return':
            return result.total_return
        elif self.scoring_metric == 'profit_factor':
            return result.profit_factor
        elif self.scoring_metric == 'win_rate':
            return result.win_rate
        elif self.scoring_metric == 'custom':
            # 커스텀 점수 계산 (샤프 비율 * 승률 * (1 - 최대 낙폭))
            return result.sharpe_ratio * result.win_rate * (1 - result.max_drawdown)
        else:
            raise ValueError(f"지원하지 않는 점수 메트릭: {self.scoring_metric}")
    
    def plot_optimization_results(self, result: OptimizationResult) -> Dict[str, Any]:
        """최적화 결과 시각화"""
        try:
            # 파라미터 중요도 플롯
            param_importance = self._calculate_param_importance(result)
            
            # 성과 메트릭스 플롯
            metrics_plot = self._plot_metrics(result)
            
            # 최적화 과정 플롯
            optimization_plot = self._plot_optimization_process(result)
            
            return {
                'param_importance': param_importance,
                'metrics': metrics_plot,
                'optimization_process': optimization_plot
            }
            
        except Exception as e:
            self.logger.error(f"최적화 결과 시각화 중 오류 발생: {str(e)}")
            raise
    
    def _calculate_param_importance(self, result: OptimizationResult) -> Dict[str, float]:
        """파라미터 중요도 계산"""
        importance = {}
        for param in self.param_grid.keys():
            # 파라미터 값과 점수의 상관관계 계산
            values = [score['params'][param] for score in result.param_scores]
            scores = [score['score'] for score in result.param_scores]
            correlation = np.corrcoef(values, scores)[0, 1]
            importance[param] = abs(correlation)
        return importance
    
    def _plot_metrics(self, result: OptimizationResult) -> Dict[str, Any]:
        """성과 메트릭스 플롯"""
        metrics = result.performance_metrics
        return {
            'total_return': metrics['total_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor']
        }
    
    def _plot_optimization_process(self, result: OptimizationResult) -> pd.DataFrame:
        """최적화 과정 플롯"""
        return result.optimization_history
    
    def save_results(self, result: OptimizationResult, directory: str):
        """최적화 결과 저장"""
        try:
            # 결과 디렉토리 생성
            import os
            os.makedirs(directory, exist_ok=True)
            
            # 최적 파라미터 저장
            with open(f"{directory}/best_params.json", 'w') as f:
                import json
                json.dump(result.best_params, f, indent=4)
            
            # 파라미터 점수 저장
            pd.DataFrame(result.param_scores).to_csv(
                f"{directory}/param_scores.csv",
                index=False
            )
            
            # 최적화 과정 저장
            result.optimization_history.to_csv(
                f"{directory}/optimization_history.csv",
                index=False
            )
            
            # 성과 메트릭스 저장
            pd.DataFrame([result.performance_metrics]).to_csv(
                f"{directory}/performance_metrics.csv",
                index=False
            )
            
        except Exception as e:
            self.logger.error(f"최적화 결과 저장 중 오류 발생: {str(e)}")
            raise 