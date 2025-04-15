import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import time
from datetime import datetime
import os
from itertools import product
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import Parallel, delayed

from .model_factory import BayesianModelFactory

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    베이지안 시계열 모델의 하이퍼파라미터 최적화를 위한 클래스
    
    그리드 서치나 랜덤 서치를 사용하여 최적의 하이퍼파라미터를 찾습니다.
    """
    
    def __init__(self, 
                 model_type: str = "ar",
                 search_method: str = "grid",
                 cv_folds: int = 3,
                 metric: str = "rmse",
                 n_jobs: int = 1,
                 verbose: bool = True,
                 random_state: Optional[int] = None):
        """
        하이퍼파라미터 최적화 클래스 초기화
        
        Args:
            model_type: 모델 유형 ('ar', 'gp', 'structural')
            search_method: 검색 방법 ('grid', 'random')
            cv_folds: 교차 검증 폴드 수
            metric: 평가 지표 ('rmse', 'mae', 'mape')
            n_jobs: 병렬 처리를 위한 작업 수
            verbose: 상세 출력 여부
            random_state: 랜덤 시드
        """
        self.model_type = model_type
        self.search_method = search_method
        self.cv_folds = cv_folds
        self.metric = metric
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        
        # 랜덤 시드 설정
        if random_state is not None:
            np.random.seed(random_state)
        
        # 최적화 결과 저장 변수
        self.results_ = []
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        
        # 평가 지표 함수 설정
        if metric == "rmse":
            self.eval_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "mae":
            self.eval_func = lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)
        elif metric == "mape":
            self.eval_func = lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        else:
            logger.warning(f"지원하지 않는 평가 지표: {metric}, 기본값 RMSE로 설정합니다.")
            self.eval_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
    
    def _get_param_grid(self, model_type: str) -> Dict[str, List[Any]]:
        """
        모델 유형에 따른 기본 하이퍼파라미터 그리드 생성
        
        Args:
            model_type: 모델 유형
            
        Returns:
            하이퍼파라미터 그리드
        """
        if model_type == "ar":
            return {
                "ar_order": [1, 3, 5, 7],
                "seasonality": [True, False],
                "num_seasons": [7, 14, 30],
                "standardize": [True, False]
            }
        elif model_type == "gp":
            return {
                "kernel_type": ["rbf", "matern32", "matern52"],
                "seasonality": [True, False],
                "period": [7, 14, 30],
                "trend": [True, False]
            }
        elif model_type == "structural":
            return {
                "level": [True],
                "trend": [True, False],
                "seasonality": [True, False],
                "season_period": [7, 14, 30],
                "damped_trend": [True, False]
            }
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {model_type}")
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]], n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        하이퍼파라미터 조합 생성
        
        Args:
            param_grid: 하이퍼파라미터 그리드
            n_samples: 랜덤 샘플링할 수 (search_method가 'random'인 경우에만 사용)
            
        Returns:
            하이퍼파라미터 조합 리스트
        """
        if self.search_method == "grid":
            # 모든 가능한 조합 생성
            keys = param_grid.keys()
            values = param_grid.values()
            combinations = [dict(zip(keys, combo)) for combo in product(*values)]
            return combinations
        
        elif self.search_method == "random":
            # 랜덤 샘플링
            if n_samples is None:
                # 그리드 검색의 조합 수의 1/3
                keys = param_grid.keys()
                values = param_grid.values()
                total_combinations = np.prod([len(v) for v in values])
                n_samples = max(1, int(total_combinations / 3))
                logger.info(f"총 가능한 조합 수: {total_combinations}, 랜덤 샘플링 수: {n_samples}")
            
            combinations = []
            for _ in range(n_samples):
                params = {}
                for key, values in param_grid.items():
                    params[key] = np.random.choice(values)
                combinations.append(params)
            
            return combinations
        
        else:
            raise ValueError(f"지원하지 않는 검색 방법: {self.search_method}")
    
    def _evaluate_model(self, params: Dict[str, Any], train_data: pd.Series, 
                      val_data: pd.Series, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 하이퍼파라미터 조합을 평가
        
        Args:
            params: 하이퍼파라미터 조합
            train_data: 학습 데이터
            val_data: 검증 데이터
            sampling_params: MCMC 샘플링 파라미터
            
        Returns:
            평가 결과 딕셔너리
        """
        try:
            # 모델 생성
            model = BayesianModelFactory.get_model(self.model_type, **params)
            
            # 모델 학습
            start_time = time.time()
            model.fit(train_data, sampling_params=sampling_params)
            train_time = time.time() - start_time
            
            # 예측 및 평가
            forecast, lower, upper = model.predict(n_forecast=len(val_data))
            
            # 메트릭스 계산
            score = self.eval_func(val_data.values, forecast)
            
            # 결과 반환
            result = {
                "params": params,
                "score": score,
                "train_time": train_time,
                "forecast": forecast,
                "lower": lower,
                "upper": upper
            }
            
            # 공간 절약을 위해 예측 결과는 저장하지 않는 옵션
            if not self.verbose:
                del result["forecast"]
                del result["lower"]
                del result["upper"]
            
            return result
        
        except Exception as e:
            logger.error(f"모델 평가 오류 - 파라미터: {params}, 오류: {str(e)}")
            return {
                "params": params,
                "score": np.inf,
                "train_time": -1,
                "error": str(e)
            }
    
    def _evaluate_fold(self, fold_id: int, train_indices: np.ndarray, val_indices: np.ndarray, 
                     data: pd.Series, params: Dict[str, Any], 
                     sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 교차 검증 폴드에서 단일 하이퍼파라미터 조합 평가
        
        Args:
            fold_id: 폴드 ID
            train_indices: 학습 인덱스
            val_indices: 검증 인덱스
            data: 전체 데이터
            params: 하이퍼파라미터 조합
            sampling_params: MCMC 샘플링 파라미터
            
        Returns:
            평가 결과 딕셔너리
        """
        train_data = data.iloc[train_indices]
        val_data = data.iloc[val_indices]
        
        result = self._evaluate_model(params, train_data, val_data, sampling_params)
        result["fold"] = fold_id
        
        return result
    
    def _perform_cross_validation(self, params: Dict[str, Any], data: pd.Series, 
                               sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        하이퍼파라미터 조합에 대한 교차 검증 수행
        
        Args:
            params: 하이퍼파라미터 조합
            data: 전체 데이터
            sampling_params: MCMC 샘플링 파라미터
            
        Returns:
            평가 결과 딕셔너리
        """
        # 데이터 분할
        n_samples = len(data)
        fold_size = n_samples // self.cv_folds
        
        fold_results = []
        
        for fold_id in range(self.cv_folds):
            # 마지막 폴드는 나머지 데이터를 모두 포함
            if fold_id == self.cv_folds - 1:
                val_start_idx = fold_id * fold_size
                val_end_idx = n_samples
            else:
                val_start_idx = fold_id * fold_size
                val_end_idx = (fold_id + 1) * fold_size
            
            val_indices = np.arange(val_start_idx, val_end_idx)
            train_indices = np.array([i for i in range(n_samples) if i not in val_indices])
            
            fold_result = self._evaluate_fold(fold_id, train_indices, val_indices, data, params, sampling_params)
            fold_results.append(fold_result)
        
        # 평균 점수 계산
        scores = [r["score"] for r in fold_results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # 결과 취합
        cv_result = {
            "params": params,
            "mean_score": mean_score,
            "std_score": std_score,
            "fold_scores": scores,
            "fold_results": fold_results if self.verbose else None
        }
        
        return cv_result
    
    def fit(self, data: pd.Series, param_grid: Optional[Dict[str, List[Any]]] = None, 
          n_samples: Optional[int] = None, sampling_params: Optional[Dict[str, Any]] = None,
          save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        하이퍼파라미터 최적화 수행
        
        Args:
            data: 시계열 데이터
            param_grid: 하이퍼파라미터 그리드 (None이면 기본 그리드 사용)
            n_samples: 랜덤 샘플링할 수 (search_method가 'random'인 경우에만 사용)
            sampling_params: MCMC 샘플링 파라미터
            save_path: 결과 저장 경로
            
        Returns:
            최적화 결과
        """
        # 기본 파라미터 그리드 생성
        if param_grid is None:
            param_grid = self._get_param_grid(self.model_type)
        
        # 기본 샘플링 파라미터
        if sampling_params is None:
            sampling_params = {
                'draws': 300,
                'tune': 300,
                'chains': 2,
                'target_accept': 0.9
            }
        
        # 하이퍼파라미터 조합 생성
        param_combinations = self._generate_param_combinations(param_grid, n_samples)
        
        logger.info(f"하이퍼파라미터 최적화 시작: 모델={self.model_type}, 방법={self.search_method}, 조합 수={len(param_combinations)}")
        
        # 진행 상황 추적 변수
        total_combinations = len(param_combinations)
        start_time = time.time()
        
        # 병렬 처리를 위한 작업 생성
        if self.n_jobs > 1:
            cv_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._perform_cross_validation)(params, data, sampling_params)
                for params in param_combinations
            )
        else:
            cv_results = []
            for i, params in enumerate(param_combinations):
                if self.verbose:
                    logger.info(f"조합 {i+1}/{total_combinations} 평가 중: {params}")
                
                cv_result = self._perform_cross_validation(params, data, sampling_params)
                
                if self.verbose:
                    logger.info(f"조합 {i+1}/{total_combinations} 결과: 평균 점수 = {cv_result['mean_score']:.4f}, 표준편차 = {cv_result['std_score']:.4f}")
                
                cv_results.append(cv_result)
        
        # 최적의 조합 찾기
        best_idx = np.argmin([r["mean_score"] for r in cv_results])
        self.best_params_ = cv_results[best_idx]["params"]
        self.best_score_ = cv_results[best_idx]["mean_score"]
        
        # 전체 결과 저장
        self.results_ = cv_results
        
        # 최종 모델 학습 (전체 데이터로)
        logger.info(f"최적 파라미터로 최종 모델 학습: {self.best_params_}")
        self.best_model_ = BayesianModelFactory.get_model(self.model_type, **self.best_params_)
        self.best_model_.fit(data, sampling_params=sampling_params)
        
        # 총 소요 시간
        total_time = time.time() - start_time
        
        # 결과 요약
        optimization_summary = {
            "model_type": self.model_type,
            "search_method": self.search_method,
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "total_combinations": total_combinations,
            "total_time": total_time,
            "cv_folds": self.cv_folds,
            "metric": self.metric
        }
        
        logger.info(f"하이퍼파라미터 최적화 완료: 최적 점수 = {self.best_score_:.4f}, 소요 시간 = {total_time:.2f}초")
        logger.info(f"최적 파라미터: {self.best_params_}")
        
        # 결과 저장
        if save_path is not None:
            self.save_results(save_path)
        
        return optimization_summary
    
    def save_results(self, save_path: str) -> None:
        """
        최적화 결과 저장
        
        Args:
            save_path: 저장 경로
        """
        # 저장 디렉토리 생성
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        # 결과 요약
        optimization_summary = {
            "model_type": self.model_type,
            "search_method": self.search_method,
            "best_params": self.best_params_,
            "best_score": float(self.best_score_),  # numpy.float32를 일반 float로 변환
            "total_combinations": len(self.results_),
            "cv_folds": self.cv_folds,
            "metric": self.metric,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 결과 저장
        with open(save_path, 'w') as f:
            json.dump(optimization_summary, f, indent=4)
        
        logger.info(f"최적화 결과 저장 완료: {save_path}")
    
    def load_results(self, load_path: str) -> Dict[str, Any]:
        """
        저장된 최적화 결과 로드
        
        Args:
            load_path: 로드 경로
            
        Returns:
            최적화 결과
        """
        with open(load_path, 'r') as f:
            optimization_summary = json.load(f)
        
        self.model_type = optimization_summary["model_type"]
        self.search_method = optimization_summary["search_method"]
        self.best_params_ = optimization_summary["best_params"]
        self.best_score_ = optimization_summary["best_score"]
        
        logger.info(f"최적화 결과 로드 완료: {load_path}")
        
        return optimization_summary
    
    def plot_optimization_results(self, top_n: int = 10, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        최적화 결과 시각화
        
        Args:
            top_n: 상위 몇 개의 결과를 표시할지 설정
            figsize: 그래프 크기
            
        Returns:
            그래프 객체
        """
        if not self.results_:
            raise ValueError("최적화 결과가 없습니다. 먼저 fit() 메서드를 호출하세요.")
        
        # 결과 정렬
        sorted_results = sorted(self.results_, key=lambda x: x["mean_score"])
        
        # 상위 N개 결과 추출
        top_results = sorted_results[:min(top_n, len(sorted_results))]
        
        # 평균 점수 및 표준편차 추출
        mean_scores = [r["mean_score"] for r in top_results]
        std_scores = [r["std_score"] for r in top_results]
        
        # 파라미터 문자열 변환
        param_strings = []
        for result in top_results:
            params = result["params"]
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            param_strings.append(param_str)
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 바 플롯
        x = np.arange(len(param_strings))
        bars = ax.bar(x, mean_scores, yerr=std_scores, alpha=0.7, capsize=5)
        
        # 최적 조합 강조
        best_idx = np.argmin(mean_scores)
        bars[best_idx].set_color('green')
        
        # 레이블 및 제목
        ax.set_ylabel(f'{self.metric.upper()}')
        ax.set_title(f'{self.model_type.upper()} 모델 하이퍼파라미터 최적화 결과 (상위 {len(param_strings)}개)')
        ax.set_xticks(x)
        ax.set_xticklabels(param_strings, rotation=45, ha='right')
        
        # 그리드
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 값 표시
        for i, v in enumerate(mean_scores):
            ax.text(i, v + std_scores[i] + 0.01 * max(mean_scores), 
                   f'{v:.4f}', ha='center', va='bottom', 
                   fontweight='bold' if i == best_idx else 'normal')
        
        plt.tight_layout()
        return fig
    
    def predict(self, n_forecast: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        최적 모델을 사용하여 예측 수행
        
        Args:
            n_forecast: 예측할 미래 시점 수
            
        Returns:
            예측값, 하한, 상한
        """
        if self.best_model_ is None:
            raise ValueError("최적화된 모델이 없습니다. 먼저 fit() 메서드를 호출하세요.")
        
        return self.best_model_.predict(n_forecast=n_forecast)
    
    def plot_forecast(self, original_data: pd.Series, n_forecast: int = 10,
                    title: Optional[str] = None, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        최적 모델의 예측 결과 시각화
        
        Args:
            original_data: 원본 시계열 데이터
            n_forecast: 예측할 미래 시점 수
            title: 그래프 제목
            figsize: 그래프 크기
            
        Returns:
            그래프 객체
        """
        if self.best_model_ is None:
            raise ValueError("최적화된 모델이 없습니다. 먼저 fit() 메서드를 호출하세요.")
        
        # 예측 수행
        forecast, lower, upper = self.predict(n_forecast=n_forecast)
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 원본 데이터 플롯
        if isinstance(original_data.index, pd.DatetimeIndex):
            ax.plot(original_data.index, original_data, label='실제 데이터', color='blue')
            
            # 미래 날짜 생성
            last_date = original_data.index[-1]
            freq = pd.infer_freq(original_data.index)
            if freq is None:
                freq = 'D'  # 기본값 일단위
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=n_forecast, 
                                       freq=freq)
            
            # 예측 결과 플롯
            ax.plot(future_dates, forecast, label='예측', color='red')
            ax.fill_between(future_dates, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
        else:
            # 날짜가 아닌 인덱스인 경우
            x = np.arange(len(original_data))
            ax.plot(x, original_data, label='실제 데이터', color='blue')
            
            future_x = np.arange(len(original_data), len(original_data) + n_forecast)
            ax.plot(future_x, forecast, label='예측', color='red')
            ax.fill_between(future_x, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
        
        # 제목 및 레이블
        if title is None:
            title = f"{self.model_type.upper()} 모델 예측 (최적 하이퍼파라미터)"
        ax.set_title(title)
        ax.set_xlabel('시간')
        ax.set_ylabel('값')
        ax.legend()
        ax.grid(True)
        
        # 최적 하이퍼파라미터 표시
        params_text = ", ".join([f"{k}={v}" for k, v in self.best_params_.items()])
        plt.figtext(0.5, 0.01, f"최적 파라미터: {params_text}", ha='center', fontsize=10, 
                   bbox={'facecolor': 'lightgrey', 'alpha': 0.5, 'pad': 5})
        
        plt.tight_layout()
        return fig 