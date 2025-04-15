import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple, Optional
import arviz as az
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class BayesianTimeSeriesPredictor:
    """
    베이지안 시계열 예측 모델 클래스
    
    다양한 베이지안 시계열 모델을 구현하고 예측을 수행하는 클래스입니다.
    """
    
    def __init__(self, model_type: str = "ar", seasonality: bool = False,
                 num_seasons: int = 7, ar_order: int = 1):
        """
        베이지안 시계열 예측 모델 초기화
        
        Args:
            model_type (str): 모델 유형 ('ar', 'gp', 'structural', 'ar_gp')
            seasonality (bool): 계절성 포함 여부
            num_seasons (int): 계절 주기
            ar_order (int): AR 모델의 차수
        """
        self.model_type = model_type
        self.seasonality = seasonality
        self.num_seasons = num_seasons
        self.ar_order = ar_order
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.y_train = None
        self.train_dates = None
        
        logger.info(f"베이지안 시계열 예측기 초기화: {model_type} 모델, 계절성: {seasonality}")
    
    def _preprocess_data(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        데이터 전처리
        
        Args:
            data: 시계열 데이터
            
        Returns:
            전처리된 데이터
        """
        if isinstance(data, pd.Series):
            data_values = data.values.reshape(-1, 1)
        else:
            data_values = data.reshape(-1, 1)
            
        return self.scaler.fit_transform(data_values).flatten()
    
    def _build_autoregressive_model(self, data: np.ndarray) -> pm.Model:
        """
        자기회귀(AR) 모델 구축
        
        Args:
            data: 전처리된 데이터
            
        Returns:
            PyMC3 모델
        """
        with pm.Model() as model:
            # 초기 값 설정
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # AR 계수
            coeffs = pm.Normal('coeffs', mu=0, sigma=0.5, shape=self.ar_order)
            
            # 계절성 처리
            if self.seasonality:
                season_effects = pm.Normal('season_effects', mu=0, sigma=0.5, shape=self.num_seasons)
                season_idx = np.arange(len(data)) % self.num_seasons
            
            # 모델 정의
            mu = pm.Normal('mu', mu=0, sigma=1)
            
            # 예측값 계산
            y_pred = tt.zeros(len(data))
            y_pred = tt.set_subtensor(y_pred[:self.ar_order], data[:self.ar_order])
            
            for i in range(self.ar_order, len(data)):
                pred = mu
                for j in range(self.ar_order):
                    pred = pred + coeffs[j] * data[i-j-1]
                
                if self.seasonality:
                    pred = pred + season_effects[season_idx[i]]
                
                y_pred = tt.set_subtensor(y_pred[i], pred)
            
            # 우도 정의
            likelihood = pm.Normal('y', mu=y_pred[self.ar_order:], 
                                   sigma=sigma, 
                                   observed=data[self.ar_order:])
        
        return model
    
    def fit(self, data: Union[pd.Series, np.ndarray], 
            dates: Optional[pd.DatetimeIndex] = None,
            sampling_params: Dict = None) -> None:
        """
        모델 학습
        
        Args:
            data: 시계열 데이터
            dates: 날짜 인덱스 (선택 사항)
            sampling_params: MCMC 샘플링 파라미터
        """
        # 기본 샘플링 파라미터
        default_params = {
            'draws': 1000,
            'tune': 1000,
            'chains': 2,
            'target_accept': 0.95,
            'return_inferencedata': True
        }
        
        if sampling_params:
            default_params.update(sampling_params)
        
        # 데이터 전처리
        processed_data = self._preprocess_data(data)
        self.y_train = processed_data
        self.train_dates = dates
        
        # 모델 유형에 따른 모델 구축
        if self.model_type == "ar":
            self.model = self._build_autoregressive_model(processed_data)
        elif self.model_type == "gp":
            # 가우시안 프로세스 모델 구현 예정
            raise NotImplementedError("가우시안 프로세스 모델은 아직 구현되지 않았습니다.")
        elif self.model_type == "structural":
            # 구조적 시계열 모델 구현 예정
            raise NotImplementedError("구조적 시계열 모델은 아직 구현되지 않았습니다.")
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {self.model_type}")
        
        # MCMC 샘플링
        with self.model:
            self.trace = pm.sample(**default_params)
        
        self.is_fitted = True
        logger.info(f"모델 학습 완료: {len(processed_data)} 데이터 포인트, {default_params['draws']} 샘플")

    def predict(self, n_forecast: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        예측 수행
        
        Args:
            n_forecast: 예측할 미래 시점 수
            
        Returns:
            예측값, 하한, 상한
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        # 예측을 위한 배열 초기화
        y_pred = np.zeros((n_forecast, len(self.trace.posterior.chain) * len(self.trace.posterior.draw)))
        
        # 마지막 학습 데이터
        last_data = self.y_train[-self.ar_order:]
        
        for i in range(n_forecast):
            # AR 모델 예측
            if self.model_type == "ar":
                for idx, chain in enumerate(self.trace.posterior.chain):
                    for jdx, draw in enumerate(self.trace.posterior.draw):
                        mu = self.trace.posterior.mu.values[idx, jdx]
                        coeffs = self.trace.posterior.coeffs.values[idx, jdx]
                        
                        # 계절 효과
                        seasonal_effect = 0
                        if self.seasonality:
                            season_idx = (len(self.y_train) + i) % self.num_seasons
                            seasonal_effect = self.trace.posterior.season_effects.values[idx, jdx, season_idx]
                        
                        # 예측값 계산
                        pred = mu
                        for j in range(self.ar_order):
                            if i - j - 1 < 0:
                                ar_term = last_data[self.ar_order + (i - j - 1)]
                            else:
                                ar_term = y_pred[i - j - 1, idx * len(self.trace.posterior.draw) + jdx]
                            pred += coeffs[j] * ar_term
                        
                        pred += seasonal_effect
                        y_pred[i, idx * len(self.trace.posterior.draw) + jdx] = pred
        
        # 사후 분포에서 통계량 계산
        y_mean = np.mean(y_pred, axis=1)
        y_lower = np.percentile(y_pred, 2.5, axis=1)
        y_upper = np.percentile(y_pred, 97.5, axis=1)
        
        # 역변환
        y_mean_orig = self.scaler.inverse_transform(y_mean.reshape(-1, 1)).flatten()
        y_lower_orig = self.scaler.inverse_transform(y_lower.reshape(-1, 1)).flatten()
        y_upper_orig = self.scaler.inverse_transform(y_upper.reshape(-1, 1)).flatten()
        
        return y_mean_orig, y_lower_orig, y_upper_orig
    
    def plot_forecast(self, original_data: Union[pd.Series, np.ndarray], 
                     forecast: np.ndarray, 
                     lower: np.ndarray, 
                     upper: np.ndarray,
                     title: str = "베이지안 시계열 예측") -> plt.Figure:
        """
        예측 결과 시각화
        
        Args:
            original_data: 원본 시계열 데이터
            forecast: 예측값
            lower: 하한
            upper: 상한
            title: 그래프 제목
            
        Returns:
            그래프 객체
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 학습 데이터 플롯
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
                
                # 예측 플롯
                ax.plot(future_dates, forecast, label='예측', color='red')
                ax.fill_between(future_dates, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
            else:
                x = np.arange(len(original_data))
                future_x = np.arange(len(original_data), len(original_data) + len(forecast))
                ax.plot(x, original_data, label='실제 데이터', color='blue')
                ax.plot(future_x, forecast, label='예측', color='red')
                ax.fill_between(future_x, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
        else:
            x = np.arange(len(original_data))
            future_x = np.arange(len(original_data), len(original_data) + len(forecast))
            
            ax.plot(x, original_data, label='실제 데이터', color='blue')
            ax.plot(future_x, forecast, label='예측', color='red')
            ax.fill_between(future_x, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
        
        ax.set_title(title)
        ax.set_xlabel('시간')
        ax.set_ylabel('값')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_trace(self, var_names: Optional[List[str]] = None) -> None:
        """
        트레이스 플롯 생성
        
        Args:
            var_names: 플롯할 변수 이름 목록
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        az.plot_trace(self.trace, var_names=var_names)
        plt.tight_layout()
    
    def evaluate(self, test_data: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        테스트 데이터로 모델 평가
        
        Args:
            test_data: 테스트 데이터
            
        Returns:
            평가 지표 딕셔너리
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        # 테스트 데이터 길이만큼 예측
        forecasts, lower, upper = self.predict(n_forecast=len(test_data))
        
        # 평가 지표 계산
        metrics = {
            'rmse': np.sqrt(mean_squared_error(test_data, forecasts)),
            'mae': mean_absolute_error(test_data, forecasts),
            'r2': r2_score(test_data, forecasts)
        }
        
        return metrics 