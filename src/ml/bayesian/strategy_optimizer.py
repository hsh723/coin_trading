import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime
import json
import os
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from .model_factory import BayesianModelFactory
from .backtesting import BayesianBacktester

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    거래 전략 최적화 도구
    
    주요 기능:
    - 전략 파라미터 최적화
    - 전략 성능 평가
    - 전략 조합 최적화
    - 전략 적응성 분석
    """
    
    def __init__(self,
                 model_type: str = "ar",
                 initial_params: Dict[str, Any] = None,
                 optimization_method: str = "differential_evolution",
                 n_jobs: int = -1,
                 save_dir: str = "./strategy_optimization"):
        """
        전략 최적화 도구 초기화
        
        Args:
            model_type: 모델 유형
            initial_params: 초기 파라미터
            optimization_method: 최적화 방법
            n_jobs: 병렬 처리 작업 수
            save_dir: 결과 저장 디렉토리
        """
        self.model_type = model_type
        self.initial_params = initial_params or {}
        self.optimization_method = optimization_method
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.save_dir = save_dir
        
        # 결과 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 백테스팅 시스템 초기화
        self.backtester = BayesianBacktester(
            model_type=model_type,
            model_params=initial_params
        )
    
    def optimize_strategy(self,
                         data: pd.DataFrame,
                         param_bounds: Dict[str, Tuple[float, float]],
                         metric: str = "sharpe_ratio",
                         n_splits: int = 5,
                         max_iter: int = 100) -> Dict[str, Any]:
        """
        전략 파라미터 최적화
        
        Args:
            data: 시계열 데이터
            param_bounds: 파라미터 범위
            metric: 최적화 메트릭
            n_splits: 시계열 분할 수
            max_iter: 최대 반복 횟수
            
        Returns:
            최적화 결과
        """
        logger.info("전략 최적화 시작...")
        
        # 시계열 분할
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        def objective(params):
            # 파라미터 변환
            param_dict = {k: v for k, v in zip(param_bounds.keys(), params)}
            
            # 교차 검증 점수 계산
            scores = []
            
            for train_idx, test_idx in tscv.split(data):
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                # 백테스팅 실행
                self.backtester.model_params = param_dict
                self.backtester.model = BayesianModelFactory.get_model(
                    model_type=self.model_type,
                    **param_dict
                )
                
                results = self.backtester.run(
                    data=pd.concat([train_data, test_data]),
                    train_size=len(train_data) / (len(train_data) + len(test_data))
                )
                
                scores.append(results[metric])
            
            # 평균 점수 반환 (최소화 문제로 변환)
            return -np.mean(scores)
        
        # 최적화 실행
        if self.optimization_method == "differential_evolution":
            result = differential_evolution(
                objective,
                bounds=list(param_bounds.values()),
                maxiter=max_iter,
                workers=self.n_jobs
            )
        else:
            result = minimize(
                objective,
                x0=[np.mean(b) for b in param_bounds.values()],
                bounds=list(param_bounds.values()),
                method='L-BFGS-B',
                options={'maxiter': max_iter}
            )
        
        # 최적 파라미터 저장
        best_params = {k: v for k, v in zip(param_bounds.keys(), result.x)}
        
        # 최적화 결과 저장
        optimization_results = {
            'best_params': best_params,
            'best_score': -result.fun,
            'optimization_method': self.optimization_method,
            'n_splits': n_splits,
            'max_iter': max_iter
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.save_dir, f"optimization_results_{timestamp}.json")
        
        with open(result_file, 'w') as f:
            json.dump(optimization_results, f, indent=4, default=str)
        
        return optimization_results
    
    def evaluate_strategy(self,
                         data: pd.DataFrame,
                         params: Dict[str, Any],
                         n_splits: int = 5) -> Dict[str, Any]:
        """
        전략 성능 평가
        
        Args:
            data: 시계열 데이터
            params: 전략 파라미터
            n_splits: 시계열 분할 수
            
        Returns:
            성능 평가 결과
        """
        logger.info("전략 성능 평가 시작...")
        
        # 시계열 분할
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # 성능 지표 저장
        metrics = {
            'total_return': [],
            'annual_return': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'win_rate': []
        }
        
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # 백테스팅 실행
            self.backtester.model_params = params
            self.backtester.model = BayesianModelFactory.get_model(
                model_type=self.model_type,
                **params
            )
            
            results = self.backtester.run(
                data=pd.concat([train_data, test_data]),
                train_size=len(train_data) / (len(train_data) + len(test_data))
            )
            
            # 성능 지표 저장
            for metric in metrics.keys():
                metrics[metric].append(results[metric])
        
        # 평균 및 표준편차 계산
        evaluation_results = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in metrics.items()
        }
        
        # 평가 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.save_dir, f"evaluation_results_{timestamp}.json")
        
        with open(result_file, 'w') as f:
            json.dump(evaluation_results, f, indent=4, default=str)
        
        return evaluation_results
    
    def optimize_strategy_combination(self,
                                    data: pd.DataFrame,
                                    strategies: List[Dict[str, Any]],
                                    metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        전략 조합 최적화
        
        Args:
            data: 시계열 데이터
            strategies: 전략 목록
            metric: 최적화 메트릭
            
        Returns:
            최적 전략 조합
        """
        logger.info("전략 조합 최적화 시작...")
        
        def objective(weights):
            # 가중치 정규화
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # 각 전략의 예측 결과 계산
            predictions = []
            
            for strategy in strategies:
                self.backtester.model_params = strategy['params']
                self.backtester.model = BayesianModelFactory.get_model(
                    model_type=strategy['model_type'],
                    **strategy['params']
                )
                
                prediction = self.backtester.model.predict(data)
                predictions.append(prediction)
            
            # 가중 평균 예측
            combined_prediction = np.average(predictions, weights=weights, axis=0)
            
            # 백테스팅 실행
            results = self.backtester.run(data)
            
            return -results[metric]
        
        # 가중치 최적화
        initial_weights = np.ones(len(strategies)) / len(strategies)
        bounds = [(0, 1) for _ in range(len(strategies))]
        
        result = minimize(
            objective,
            x0=initial_weights,
            bounds=bounds,
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            method='SLSQP'
        )
        
        # 최적 가중치 저장
        best_weights = result.x / np.sum(result.x)
        
        # 최적화 결과 저장
        combination_results = {
            'strategies': strategies,
            'weights': best_weights.tolist(),
            'score': -result.fun
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.save_dir, f"combination_results_{timestamp}.json")
        
        with open(result_file, 'w') as f:
            json.dump(combination_results, f, indent=4, default=str)
        
        return combination_results 