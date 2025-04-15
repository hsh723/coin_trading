import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
import logging

from .model_factory import BayesianModelFactory

logger = logging.getLogger(__name__)

class BayesianEnsembleModel:
    """
    베이지안 시계열 모델 앙상블 클래스
    
    여러 베이지안 시계열 모델의 예측을 결합하여 더 안정적인 예측을 제공합니다.
    """
    
    def __init__(self, 
                 models_config: List[Dict[str, Any]], 
                 ensemble_method: str = "mean",
                 weights: Optional[List[float]] = None):
        """
        베이지안 앙상블 모델 초기화
        
        Args:
            models_config: 모델 설정 리스트 (각 모델에 대한 유형 및 파라미터 설정)
            ensemble_method: 앙상블 방법 ('mean', 'weighted', 'median', 'bayes')
            weights: 가중 평균을 위한 가중치 (ensemble_method가 'weighted'인 경우에만 사용)
        """
        self.models_config = models_config
        self.ensemble_method = ensemble_method
        self.models = []
        self.model_names = []
        self.is_fitted = False
        self.scaler = StandardScaler()
        
        # 가중치 설정 (가중 평균 방식에만 적용)
        if weights is not None and ensemble_method == "weighted":
            if len(weights) != len(models_config):
                raise ValueError(f"가중치 개수({len(weights)})가 모델 개수({len(models_config)})와 일치해야 합니다.")
            if abs(sum(weights) - 1.0) > 1e-6:
                logger.warning("가중치 합이 1이 아닙니다. 자동으로 정규화합니다.")
                weights = [w / sum(weights) for w in weights]
            self.weights = weights
        else:
            if ensemble_method == "weighted":
                # 동일 가중치로 초기화
                self.weights = [1.0 / len(models_config)] * len(models_config)
            else:
                self.weights = None
        
        # 모델 초기화
        for i, config in enumerate(models_config):
            model_type = config.get("type")
            model_name = config.get("name", f"Model_{i+1}")
            model_params = config.get("params", {})
            
            self.model_names.append(model_name)
            
            try:
                model = BayesianModelFactory.get_model(model_type, **model_params)
                self.models.append(model)
                logger.info(f"모델 '{model_name}' ({model_type}) 초기화됨")
            except Exception as e:
                logger.error(f"모델 '{model_name}' ({model_type}) 초기화 실패: {str(e)}")
                raise
        
        logger.info(f"앙상블 모델 초기화 완료: {len(self.models)}개 모델, 방법: {ensemble_method}")
    
    def fit(self, data: Union[pd.Series, np.ndarray], 
            sampling_params: Optional[Dict[str, Any]] = None) -> None:
        """
        모든 앙상블 모델 학습
        
        Args:
            data: 시계열 데이터
            sampling_params: MCMC 샘플링 파라미터 (모든 모델에 공통 적용)
        """
        if len(self.models) == 0:
            raise ValueError("학습할 모델이 없습니다.")
        
        # 기본 샘플링 파라미터
        default_params = {
            'draws': 500,
            'tune': 500,
            'chains': 2,
            'target_accept': 0.95
        }
        
        if sampling_params:
            default_params.update(sampling_params)
        
        # 표준화를 위한 데이터 형태 변환
        if isinstance(data, pd.Series):
            data_values = data.values.reshape(-1, 1)
        else:
            data_values = data.reshape(-1, 1)
        
        # 스케일러 학습 (나중에 예측값을 원래 스케일로 되돌리기 위해)
        self.scaler.fit(data_values)
        
        # 모든 모델 학습
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            logger.info(f"{i+1}/{len(self.models)} '{name}' 모델 학습 중...")
            
            try:
                model.fit(data, sampling_params=default_params)
                logger.info(f"'{name}' 모델 학습 완료")
            except Exception as e:
                logger.error(f"'{name}' 모델 학습 실패: {str(e)}")
                raise
        
        self.is_fitted = True
        logger.info(f"앙상블 모델 학습 완료: {len(self.models)}개 모델")
    
    def _update_weights(self, 
                        test_data: Union[pd.Series, np.ndarray], 
                        forecasts: List[np.ndarray]) -> None:
        """
        예측 성능에 따라 모델 가중치 업데이트
        
        Args:
            test_data: 테스트 데이터
            forecasts: 각 모델의 예측값 리스트
        """
        if self.ensemble_method != "weighted":
            return
        
        # 각 모델의 예측 오차 계산
        errors = []
        for forecast in forecasts:
            # MSE 계산
            mse = np.mean((test_data - forecast) ** 2)
            errors.append(mse)
        
        # 오차의 역수를 가중치로 사용 (오차가 적을수록 가중치 증가)
        inv_errors = [1.0 / (e + 1e-10) for e in errors]  # 0으로 나누기 방지
        
        # 가중치 정규화
        total = sum(inv_errors)
        self.weights = [e / total for e in inv_errors]
        
        # 결과 로깅
        for name, weight, error in zip(self.model_names, self.weights, errors):
            logger.info(f"모델 '{name}': 가중치 = {weight:.4f}, MSE = {error:.4f}")
    
    def predict(self, n_forecast: int = 10, 
                adaptive_weights: bool = False,
                test_data: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        앙상블 모델을 사용한 예측 수행
        
        Args:
            n_forecast: 예측할 미래 시점 수
            adaptive_weights: 예측 성능에 따라 가중치 자동 조정 여부
            test_data: 성능 평가 및 가중치 조정을 위한 테스트 데이터
                     (adaptive_weights=True인 경우 필요)
            
        Returns:
            예측값, 하한, 상한 (평균, 2.5 백분위수, 97.5 백분위수)
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        # 각 모델의 예측 수행
        forecasts = []
        lower_bounds = []
        upper_bounds = []
        
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            logger.info(f"{i+1}/{len(self.models)} '{name}' 모델 예측 중...")
            
            try:
                forecast, lower, upper = model.predict(n_forecast=n_forecast)
                forecasts.append(forecast)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
                logger.info(f"'{name}' 모델 예측 완료")
            except Exception as e:
                logger.error(f"'{name}' 모델 예측 실패: {str(e)}")
                raise
        
        # 적응형 가중치 업데이트 (필요한 경우)
        if adaptive_weights and test_data is not None and self.ensemble_method == "weighted":
            logger.info("예측 성능에 따른 가중치 업데이트 중...")
            
            # 테스트 데이터용 예측 수행
            test_forecasts = []
            for i, (model, name) in enumerate(zip(self.models, self.model_names)):
                test_forecast, _, _ = model.predict(n_forecast=len(test_data))
                test_forecasts.append(test_forecast)
            
            # 가중치 업데이트
            self._update_weights(test_data, test_forecasts)
        
        # 앙상블 방법에 따른 예측 결합
        if self.ensemble_method == "mean":
            # 단순 평균
            ensemble_forecast = np.mean(forecasts, axis=0)
            ensemble_lower = np.mean(lower_bounds, axis=0)
            ensemble_upper = np.mean(upper_bounds, axis=0)
            
        elif self.ensemble_method == "weighted":
            # 가중 평균
            ensemble_forecast = np.zeros(n_forecast)
            ensemble_lower = np.zeros(n_forecast)
            ensemble_upper = np.zeros(n_forecast)
            
            for i, (forecast, lower, upper) in enumerate(zip(forecasts, lower_bounds, upper_bounds)):
                weight = self.weights[i]
                ensemble_forecast += weight * forecast
                ensemble_lower += weight * lower
                ensemble_upper += weight * upper
                
        elif self.ensemble_method == "median":
            # 중앙값
            ensemble_forecast = np.median(forecasts, axis=0)
            ensemble_lower = np.median(lower_bounds, axis=0)
            ensemble_upper = np.median(upper_bounds, axis=0)
            
        elif self.ensemble_method == "bayes":
            # 베이지안 모델 평균 (각 시점에서 예측 분포를 혼합)
            # 간소화를 위해 평균과 표준편차를 결합하는 방법 사용
            
            # 각 모델의 하한/상한에서 표준편차 추정
            stds = []
            for lower, upper in zip(lower_bounds, upper_bounds):
                # 95% 신뢰구간에서 표준편차 추정 (상한 - 하한) / 3.92
                std = (upper - lower) / 3.92
                stds.append(std)
            
            # 각 모델의 정밀도(precision, 분산의 역수) 계산
            precisions = [1.0 / (std ** 2 + 1e-10) for std in stds]  # 0으로 나누기 방지
            total_precision = np.sum(precisions, axis=0)
            
            # 베이지안 평균 계산 (정밀도로 가중된 평균)
            ensemble_forecast = np.zeros(n_forecast)
            for i, forecast in enumerate(forecasts):
                ensemble_forecast += (precisions[i] / total_precision) * forecast
            
            # 새로운 분산 = 1 / 총 정밀도
            ensemble_variance = 1.0 / total_precision
            ensemble_std = np.sqrt(ensemble_variance)
            
            # 신뢰구간 계산
            ensemble_lower = ensemble_forecast - 1.96 * ensemble_std
            ensemble_upper = ensemble_forecast + 1.96 * ensemble_std
            
        else:
            raise ValueError(f"지원하지 않는 앙상블 방법: {self.ensemble_method}")
        
        return ensemble_forecast, ensemble_lower, ensemble_upper
    
    def plot_forecast(self, original_data: Union[pd.Series, np.ndarray],
                     forecast: np.ndarray, 
                     lower: np.ndarray, 
                     upper: np.ndarray,
                     title: str = "베이지안 앙상블 모델 예측",
                     show_individual: bool = False) -> plt.Figure:
        """
        예측 결과 시각화
        
        Args:
            original_data: 원본 시계열 데이터
            forecast: 예측값
            lower: 하한
            upper: 상한
            title: 그래프 제목
            show_individual: 개별 모델 예측 표시 여부
            
        Returns:
            그래프 객체
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 원본 데이터 플롯
        if isinstance(original_data, pd.Series) and original_data.index.dtype == 'datetime64[ns]':
            dates = original_data.index
            ax.plot(dates, original_data, label='실제 데이터', color='blue')
            
            # 예측 날짜 생성
            last_date = dates[-1]
            if isinstance(last_date, pd.Timestamp):
                freq = pd.infer_freq(dates)
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                            periods=len(forecast), 
                                            freq=freq)
                
                # 앙상블 예측 플롯
                ax.plot(future_dates, forecast, label='앙상블 예측', color='red', linewidth=2)
                ax.fill_between(future_dates, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
                
                # 개별 모델 예측 표시 (설정된 경우)
                if show_individual and hasattr(self, 'models') and len(self.models) > 0:
                    colors = ['green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
                    for i, (model, name) in enumerate(zip(self.models, self.model_names)):
                        if i < len(colors):  # 색상 제한
                            try:
                                model_forecast, _, _ = model.predict(n_forecast=len(forecast))
                                ax.plot(future_dates, model_forecast, '--', 
                                       label=f'{name} 예측', 
                                       color=colors[i], 
                                       alpha=0.7,
                                       linewidth=1)
                            except Exception as e:
                                logger.warning(f"개별 모델 '{name}' 예측 플롯 실패: {str(e)}")
            else:
                x = np.arange(len(original_data))
                future_x = np.arange(len(original_data), len(original_data) + len(forecast))
                ax.plot(x, original_data, label='실제 데이터', color='blue')
                ax.plot(future_x, forecast, label='앙상블 예측', color='red')
                ax.fill_between(future_x, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
        else:
            x = np.arange(len(original_data))
            future_x = np.arange(len(original_data), len(original_data) + len(forecast))
            
            ax.plot(x, original_data, label='실제 데이터', color='blue')
            ax.plot(future_x, forecast, label='앙상블 예측', color='red', linewidth=2)
            ax.fill_between(future_x, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
            
            # 개별 모델 예측 표시 (설정된 경우)
            if show_individual and hasattr(self, 'models') and len(self.models) > 0:
                colors = ['green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
                for i, (model, name) in enumerate(zip(self.models, self.model_names)):
                    if i < len(colors):  # 색상 제한
                        try:
                            model_forecast, _, _ = model.predict(n_forecast=len(forecast))
                            ax.plot(future_x, model_forecast, '--', 
                                   label=f'{name} 예측', 
                                   color=colors[i], 
                                   alpha=0.7,
                                   linewidth=1)
                        except Exception as e:
                            logger.warning(f"개별 모델 '{name}' 예측 플롯 실패: {str(e)}")
        
        ax.set_title(title)
        ax.set_xlabel('시간')
        ax.set_ylabel('값')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def evaluate(self, test_data: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        테스트 데이터로 앙상블 모델 평가
        
        Args:
            test_data: 테스트 데이터
            
        Returns:
            평가 지표 딕셔너리
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        # 테스트 데이터 길이만큼 예측
        forecast, lower, upper = self.predict(n_forecast=len(test_data))
        
        # 테스트 데이터가 pd.Series인 경우 값 추출
        if isinstance(test_data, pd.Series):
            test_data = test_data.values
        
        # 평가 지표 계산
        mse = np.mean((test_data - forecast) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_data - forecast))
        mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        # 개별 모델 성능 평가 (참고용)
        individual_metrics = {}
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            try:
                model_forecast, _, _ = model.predict(n_forecast=len(test_data))
                model_mse = np.mean((test_data - model_forecast) ** 2)
                model_rmse = np.sqrt(model_mse)
                model_mae = np.mean(np.abs(test_data - model_forecast))
                
                individual_metrics[name] = {
                    'rmse': model_rmse,
                    'mae': model_mae
                }
                
                logger.info(f"모델 '{name}' 평가: RMSE = {model_rmse:.4f}, MAE = {model_mae:.4f}")
            except Exception as e:
                logger.warning(f"모델 '{name}' 평가 실패: {str(e)}")
        
        # 앙상블 vs 개별 모델 성능 비교
        logger.info(f"앙상블 모델 평가: RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.2f}%")
        
        # 개별 지표 딕셔너리에 추가
        metrics['individual'] = individual_metrics
        
        return metrics 