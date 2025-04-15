import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from typing import Union, Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class GaussianProcessModel:
    """
    베이지안 가우시안 프로세스 모델 클래스
    
    시계열 데이터 예측을 위한 베이지안 가우시안 프로세스 모델입니다.
    """
    
    def __init__(self, kernel_type: str = "rbf", seasonality: bool = False, 
                 period: int = 7, trend: bool = True):
        """
        베이지안 가우시안 프로세스 모델 초기화
        
        Args:
            kernel_type (str): 커널 유형 ('rbf', 'matern32', 'matern52', 'exponential', 'periodic')
            seasonality (bool): 계절성 포함 여부
            period (int): 계절성 주기
            trend (bool): 선형 추세 포함 여부
        """
        self.kernel_type = kernel_type
        self.seasonality = seasonality
        self.period = period
        self.trend = trend
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.X_train = None
        self.y_train = None
        
        logger.info(f"가우시안 프로세스 모델 초기화: {kernel_type} 커널, 계절성: {seasonality}, 추세: {trend}")
    
    def _preprocess_data(self, data: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 전처리
        
        Args:
            data: 시계열 데이터
            
        Returns:
            X_train, y_train (전처리된 데이터)
        """
        if isinstance(data, pd.Series):
            y_values = data.values.reshape(-1, 1)
            x_values = np.arange(len(data)).reshape(-1, 1)
        else:
            y_values = data.reshape(-1, 1)
            x_values = np.arange(len(data)).reshape(-1, 1)
        
        # 데이터 정규화
        y_scaled = self.scaler.fit_transform(y_values).flatten()
        x_scaled = x_values / np.max(x_values)  # 0~1 범위로 정규화
        
        return x_scaled.flatten(), y_scaled
    
    def _build_gp_model(self, X: np.ndarray, y: np.ndarray) -> pm.Model:
        """
        가우시안 프로세스 모델 구축
        
        Args:
            X: 시간 데이터
            y: 시계열 값
            
        Returns:
            PyMC3 모델
        """
        with pm.Model() as model:
            # 노이즈 분산
            σ = pm.HalfNormal('σ', sigma=1)
            
            # 커널 파라미터
            ℓ = pm.Gamma('ℓ', alpha=2, beta=1)  # 길이 스케일
            η = pm.HalfNormal('η', sigma=1)     # 출력 스케일
            
            # 커널 선택
            if self.kernel_type == "rbf":
                # RBF(Radial Basis Function) 커널
                cov_func = η**2 * pm.gp.cov.ExpQuad(1, ℓ)
            elif self.kernel_type == "matern32":
                # Matérn 3/2 커널
                cov_func = η**2 * pm.gp.cov.Matern32(1, ℓ)
            elif self.kernel_type == "matern52":
                # Matérn 5/2 커널
                cov_func = η**2 * pm.gp.cov.Matern52(1, ℓ)
            elif self.kernel_type == "exponential":
                # 지수 커널
                cov_func = η**2 * pm.gp.cov.Exponential(1, ℓ)
            elif self.kernel_type == "periodic":
                # 주기적 커널
                period = pm.HalfNormal('period', sigma=1)
                cov_func = η**2 * pm.gp.cov.Periodic(1, period, ℓ)
            else:
                raise ValueError(f"지원하지 않는 커널 유형: {self.kernel_type}")
                
            # 계절성 커널 추가
            if self.seasonality:
                # 계절성 커널 파라미터
                ℓ_season = pm.Gamma('ℓ_season', alpha=2, beta=1)
                η_season = pm.HalfNormal('η_season', sigma=1)
                
                # 주기적 커널로 계절성 모델링
                seasonal_kernel = η_season**2 * pm.gp.cov.Periodic(1, self.period, ℓ_season)
                cov_func = cov_func + seasonal_kernel
            
            # 추세 추가
            if self.trend:
                # 선형 추세 파라미터
                c = pm.Normal('c', mu=0, sigma=5)  # 상수항
                m = pm.Normal('m', mu=0, sigma=1)  # 기울기
                
                # 평균 함수로 선형 추세 모델링
                mean_func = pm.gp.mean.Linear(coeffs=[m], intercept=c)
            else:
                # 상수 평균 함수
                c = pm.Normal('c', mu=0, sigma=1)
                mean_func = pm.gp.mean.Constant(c=c)
            
            # 가우시안 프로세스 모델 생성
            gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
            
            # 관측치에 대한 우도 정의
            gp.marginal_likelihood('y', X=X.reshape(-1, 1), y=y, noise=σ)
        
        return model
    
    def fit(self, data: Union[pd.Series, np.ndarray], 
            sampling_params: Optional[Dict] = None) -> None:
        """
        모델 학습
        
        Args:
            data: 시계열 데이터
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
        X_train, y_train = self._preprocess_data(data)
        self.X_train = X_train
        self.y_train = y_train
        
        # 가우시안 프로세스 모델 구축
        self.model = self._build_gp_model(X_train, y_train)
        
        # MCMC 샘플링
        with self.model:
            self.trace = pm.sample(**default_params)
        
        self.is_fitted = True
        logger.info(f"모델 학습 완료: {len(y_train)} 데이터 포인트, {default_params['draws']} 샘플")
    
    def predict(self, n_forecast: int = 10, 
                n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        예측 수행
        
        Args:
            n_forecast: 예측할 미래 시점 수
            n_samples: 사후 예측 샘플 수
            
        Returns:
            예측값, 하한, 상한
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        # 예측 시점
        X_new = np.linspace(
            self.X_train[-1], 
            self.X_train[-1] + n_forecast / len(self.X_train), 
            n_forecast
        )
        
        # 모델 파라미터 추출을 위한 배열 초기화
        y_pred_samples = np.zeros((n_samples, n_forecast))
        
        # 사후 샘플에서 예측
        with self.model:
            for i in range(n_samples):
                # 무작위 샘플 선택
                chain_idx = np.random.randint(0, len(self.trace.posterior.chain))
                draw_idx = np.random.randint(0, len(self.trace.posterior.draw))
                
                # 커널 파라미터 추출
                ℓ_val = float(self.trace.posterior.ℓ.values[chain_idx, draw_idx])
                η_val = float(self.trace.posterior.η.values[chain_idx, draw_idx])
                σ_val = float(self.trace.posterior.σ.values[chain_idx, draw_idx])
                
                # 커널 생성
                if self.kernel_type == "rbf":
                    cov_func = η_val**2 * pm.gp.cov.ExpQuad(1, ℓ_val)
                elif self.kernel_type == "matern32":
                    cov_func = η_val**2 * pm.gp.cov.Matern32(1, ℓ_val)
                elif self.kernel_type == "matern52":
                    cov_func = η_val**2 * pm.gp.cov.Matern52(1, ℓ_val)
                elif self.kernel_type == "exponential":
                    cov_func = η_val**2 * pm.gp.cov.Exponential(1, ℓ_val)
                elif self.kernel_type == "periodic":
                    period_val = float(self.trace.posterior.period.values[chain_idx, draw_idx])
                    cov_func = η_val**2 * pm.gp.cov.Periodic(1, period_val, ℓ_val)
                
                # 계절성 추가
                if self.seasonality:
                    ℓ_season_val = float(self.trace.posterior.ℓ_season.values[chain_idx, draw_idx])
                    η_season_val = float(self.trace.posterior.η_season.values[chain_idx, draw_idx])
                    seasonal_kernel = η_season_val**2 * pm.gp.cov.Periodic(1, self.period, ℓ_season_val)
                    cov_func = cov_func + seasonal_kernel
                
                # 평균 함수
                if self.trend:
                    m_val = float(self.trace.posterior.m.values[chain_idx, draw_idx])
                    c_val = float(self.trace.posterior.c.values[chain_idx, draw_idx])
                    mean_func = pm.gp.mean.Linear(coeffs=[m_val], intercept=c_val)
                else:
                    c_val = float(self.trace.posterior.c.values[chain_idx, draw_idx])
                    mean_func = pm.gp.mean.Constant(c=c_val)
                
                # 가우시안 프로세스 예측
                gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
                
                # 조건부 분포 계산
                mu, var = gp.predict(X_new.reshape(-1, 1), 
                                    Xnew=X_new.reshape(-1, 1), 
                                    y=self.y_train, 
                                    X=self.X_train.reshape(-1, 1),
                                    noise=σ_val, 
                                    pred_noise=False)
                
                # 조건부 분포에서 샘플링
                y_sample = np.random.normal(mu, np.sqrt(var))
                y_pred_samples[i] = y_sample
        
        # 사후 분포에서 통계량 계산
        y_mean = np.mean(y_pred_samples, axis=0)
        y_lower = np.percentile(y_pred_samples, 2.5, axis=0)
        y_upper = np.percentile(y_pred_samples, 97.5, axis=0)
        
        # 역변환
        y_mean_orig = self.scaler.inverse_transform(y_mean.reshape(-1, 1)).flatten()
        y_lower_orig = self.scaler.inverse_transform(y_lower.reshape(-1, 1)).flatten()
        y_upper_orig = self.scaler.inverse_transform(y_upper.reshape(-1, 1)).flatten()
        
        return y_mean_orig, y_lower_orig, y_upper_orig
    
    def plot_prediction(self, data: Union[pd.Series, np.ndarray], 
                     forecast: np.ndarray, 
                     lower: np.ndarray, 
                     upper: np.ndarray,
                     title: str = "가우시안 프로세스 예측") -> plt.Figure:
        """
        예측 결과 시각화
        
        Args:
            data: 원본 시계열 데이터
            forecast: 예측값
            lower: 하한
            upper: 상한
            title: 그래프 제목
            
        Returns:
            그래프 객체
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 학습 데이터 플롯
        if isinstance(data, pd.Series) and data.index.dtype == 'datetime64[ns]':
            dates = data.index
            ax.plot(dates, data, 'o-', label='실제 데이터', color='blue', markersize=4, alpha=0.7)
            
            # 예측 날짜 생성
            last_date = dates[-1]
            if isinstance(last_date, pd.Timestamp):
                freq = pd.infer_freq(dates)
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                           periods=len(forecast), 
                                           freq=freq)
                
                # 예측 플롯
                ax.plot(future_dates, forecast, 'o-', label='예측', color='red', markersize=4)
                ax.fill_between(future_dates, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
            else:
                x = np.arange(len(data))
                future_x = np.arange(len(data), len(data) + len(forecast))
                ax.plot(x, data, 'o-', label='실제 데이터', color='blue', markersize=4, alpha=0.7)
                ax.plot(future_x, forecast, 'o-', label='예측', color='red', markersize=4)
                ax.fill_between(future_x, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
        else:
            x = np.arange(len(data))
            future_x = np.arange(len(data), len(data) + len(forecast))
            
            ax.plot(x, data, 'o-', label='실제 데이터', color='blue', markersize=4, alpha=0.7)
            ax.plot(future_x, forecast, 'o-', label='예측', color='red', markersize=4)
            ax.fill_between(future_x, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
        
        ax.set_title(title)
        ax.set_xlabel('시간')
        ax.set_ylabel('값')
        ax.legend()
        ax.grid(True)
        
        return fig 