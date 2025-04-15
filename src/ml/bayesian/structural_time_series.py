import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from typing import Union, Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
import logging
import arviz as az

logger = logging.getLogger(__name__)

class StructuralTimeSeriesModel:
    """
    베이지안 구조적 시계열 모델 클래스
    
    수준, 추세, 계절성 요소를 가진 베이지안 구조적 시계열 모델입니다.
    """
    
    def __init__(self, level: bool = True, trend: bool = True, 
                 seasonality: bool = False, season_period: int = 7,
                 damped_trend: bool = False):
        """
        베이지안 구조적 시계열 모델 초기화
        
        Args:
            level (bool): 수준(레벨) 컴포넌트 포함 여부
            trend (bool): 추세 컴포넌트 포함 여부
            seasonality (bool): 계절성 컴포넌트 포함 여부
            season_period (int): 계절성 주기
            damped_trend (bool): 감쇠 추세 사용 여부
        """
        self.level = level
        self.trend = trend
        self.seasonality = seasonality
        self.season_period = season_period
        self.damped_trend = damped_trend
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.y_train = None
        self.components = {}
        
        logger.info(f"구조적 시계열 모델 초기화: 수준={level}, 추세={trend}, 계절성={seasonality}")
    
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
    
    def _build_structural_model(self, data: np.ndarray) -> pm.Model:
        """
        구조적 시계열 모델 구축
        
        Args:
            data: 전처리된 데이터
            
        Returns:
            PyMC3 모델
        """
        with pm.Model() as model:
            # 관측 노이즈
            σ_obs = pm.HalfNormal('σ_obs', sigma=0.1)
            
            # 데이터 길이
            n = len(data)
            
            # =====================
            # 수준(레벨) 컴포넌트
            # =====================
            if self.level:
                # 초기 수준
                level_0 = pm.Normal('level_0', mu=0, sigma=1)
                
                # 수준 노이즈
                σ_level = pm.HalfNormal('σ_level', sigma=0.1)
                
                # 수준 프로세스
                level_innovations = pm.Normal('level_innovations', mu=0, sigma=σ_level, shape=n-1)
                
                # 누적 수준 계산
                level = tt.zeros(n)
                level = tt.set_subtensor(level[0], level_0)
                
                if self.trend:
                    # =====================
                    # 추세 컴포넌트
                    # =====================
                    # 초기 추세
                    trend_0 = pm.Normal('trend_0', mu=0, sigma=0.1)
                    
                    # 추세 노이즈
                    σ_trend = pm.HalfNormal('σ_trend', sigma=0.01)
                    
                    # 감쇠 파라미터 (추세의 영향을 감소시킴)
                    if self.damped_trend:
                        damping_factor = pm.Beta('damping_factor', alpha=2, beta=2)
                    else:
                        damping_factor = 1.0
                    
                    # 추세 프로세스
                    trend_innovations = pm.Normal('trend_innovations', mu=0, sigma=σ_trend, shape=n-1)
                    
                    # 누적 추세 계산
                    trend = tt.zeros(n)
                    trend = tt.set_subtensor(trend[0], trend_0)
                    
                    # 수준과 추세를 함께 업데이트
                    for t in range(1, n):
                        new_trend = damping_factor * trend[t-1] + trend_innovations[t-1]
                        new_level = level[t-1] + trend[t-1] + level_innovations[t-1]
                        
                        trend = tt.set_subtensor(trend[t], new_trend)
                        level = tt.set_subtensor(level[t], new_level)
                else:
                    # 추세가 없는 경우 수준만 업데이트
                    for t in range(1, n):
                        new_level = level[t-1] + level_innovations[t-1]
                        level = tt.set_subtensor(level[t], new_level)
            else:
                # 수준이 없는 경우 (고정된 수준)
                level = 0
                
                if self.trend:
                    # 초기 추세
                    trend_0 = pm.Normal('trend_0', mu=0, sigma=0.5)
                    
                    # 추세 노이즈
                    σ_trend = pm.HalfNormal('σ_trend', sigma=0.1)
                    
                    # 추세 프로세스
                    trend_innovations = pm.Normal('trend_innovations', mu=0, sigma=σ_trend, shape=n-1)
                    
                    # 누적 추세 계산
                    trend = tt.zeros(n)
                    trend = tt.set_subtensor(trend[0], trend_0)
                    
                    for t in range(1, n):
                        new_trend = trend[t-1] + trend_innovations[t-1]
                        trend = tt.set_subtensor(trend[t], new_trend)
                else:
                    trend = 0
            
            # =====================
            # 계절성 컴포넌트
            # =====================
            if self.seasonality:
                # 계절성 노이즈
                σ_seasonal = pm.HalfNormal('σ_seasonal', sigma=0.1)
                
                # 초기 계절성 요소
                initial_seasonal = pm.Normal('initial_seasonal', mu=0, sigma=0.1, 
                                             shape=self.season_period-1)
                
                # 계절성 효과가 합이 0이 되도록 제약
                seasonal = tt.zeros(n)
                
                # 첫 주기의 계절성 설정
                for t in range(self.season_period-1):
                    if t < n:
                        seasonal = tt.set_subtensor(seasonal[t], initial_seasonal[t])
                
                # 마지막 초기 계절성 요소는 다른 요소들의 합의 음수값
                if self.season_period-1 < n:
                    seasonal = tt.set_subtensor(
                        seasonal[self.season_period-1], 
                        -tt.sum(initial_seasonal)
                    )
                
                # 계절성 혁신
                seasonal_innovations = pm.Normal('seasonal_innovations', mu=0, 
                                                 sigma=σ_seasonal, 
                                                 shape=max(0, n-self.season_period))
                
                # 나머지 계절성 요소 갱신
                for t in range(self.season_period, n):
                    # 기존 계절성 패턴 + 혁신
                    seasonal_component = seasonal[t-self.season_period] + seasonal_innovations[t-self.season_period]
                    seasonal = tt.set_subtensor(seasonal[t], seasonal_component)
            else:
                seasonal = 0
            
            # 모든 컴포넌트 합하여 예측값 생성
            prediction = level + trend + seasonal
            
            # 관측값 모델링
            obs = pm.Normal('obs', mu=prediction, sigma=σ_obs, observed=data)
            
            # 컴포넌트 저장
            self.components = {
                'level': level if self.level else None,
                'trend': trend if self.trend else None,
                'seasonal': seasonal if self.seasonality else None
            }
            
        return model
    
    def fit(self, data: Union[pd.Series, np.ndarray], sampling_params: Dict = None) -> None:
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
        processed_data = self._preprocess_data(data)
        self.y_train = processed_data
        
        # 구조적 시계열 모델 구축
        self.model = self._build_structural_model(processed_data)
        
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
        
        # 예측 결과를 저장할 배열
        y_preds = np.zeros((len(self.trace.posterior.chain) * len(self.trace.posterior.draw), n_forecast))
        
        # 체인 및 드로우 개수
        n_chains = len(self.trace.posterior.chain)
        n_draws = len(self.trace.posterior.draw)
        
        # 각 사후 샘플에 대해 예측 수행
        for c in range(n_chains):
            for d in range(n_draws):
                # 마지막 상태 가져오기
                last_data_point = len(self.y_train) - 1
                
                # 컴포넌트 초기화
                if self.level:
                    if hasattr(self.trace.posterior, 'level'):
                        last_level = self.trace.posterior.level.values[c, d, last_data_point]
                    else:
                        last_level = 0
                else:
                    last_level = 0
                
                if self.trend:
                    if hasattr(self.trace.posterior, 'trend'):
                        last_trend = self.trace.posterior.trend.values[c, d, last_data_point]
                    else:
                        last_trend = 0
                        
                    if self.damped_trend and hasattr(self.trace.posterior, 'damping_factor'):
                        damping = self.trace.posterior.damping_factor.values[c, d]
                    else:
                        damping = 1.0
                else:
                    last_trend = 0
                    damping = 1.0
                
                # 계절성 초기화
                if self.seasonality:
                    seasonal_components = np.zeros(n_forecast)
                    
                    # 마지막 계절 주기 가져오기
                    for i in range(n_forecast):
                        season_idx = (last_data_point + i + 1) % self.season_period
                        history_idx = last_data_point - (self.season_period - season_idx)
                        
                        if history_idx >= 0 and hasattr(self.trace.posterior, 'seasonal'):
                            seasonal_components[i] = self.trace.posterior.seasonal.values[c, d, history_idx]
                
                # 예측 수행
                forecasts = np.zeros(n_forecast)
                for i in range(n_forecast):
                    # 컴포넌트 갱신
                    if i > 0:
                        if self.trend:
                            last_trend = last_trend * damping
                        
                        if self.level:
                            last_level = last_level + last_trend
                    
                    # 예측값 계산
                    forecast = last_level
                    if self.trend:
                        forecast += last_trend
                    
                    if self.seasonality:
                        forecast += seasonal_components[i]
                    
                    forecasts[i] = forecast
                
                # 예측값 저장
                y_preds[c * n_draws + d] = forecasts
        
        # 사후 분포에서 통계량 계산
        y_mean = np.mean(y_preds, axis=0)
        y_lower = np.percentile(y_preds, 2.5, axis=0)
        y_upper = np.percentile(y_preds, 97.5, axis=0)
        
        # 스케일 복원
        y_mean_orig = self.scaler.inverse_transform(y_mean.reshape(-1, 1)).flatten()
        y_lower_orig = self.scaler.inverse_transform(y_lower.reshape(-1, 1)).flatten()
        y_upper_orig = self.scaler.inverse_transform(y_upper.reshape(-1, 1)).flatten()
        
        return y_mean_orig, y_lower_orig, y_upper_orig
    
    def plot_components(self) -> plt.Figure:
        """
        모델 컴포넌트 시각화
        
        Returns:
            그래프 객체
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        # 그래프 설정
        n_components = sum([1 for c in [self.level, self.trend, self.seasonality] if c])
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 3 * n_components), sharex=True)
        
        if n_components == 1:
            axes = [axes]
        
        ax_idx = 0
        
        # 수준 컴포넌트 플롯
        if self.level:
            level_mean = self.trace.posterior.level.mean(dim=["chain", "draw"]).values
            level_hdi = az.hdi(self.trace.posterior.level)
            
            axes[ax_idx].plot(level_mean, label='수준 평균')
            axes[ax_idx].fill_between(
                np.arange(len(level_mean)),
                level_hdi.sel(hdi='lower').values,
                level_hdi.sel(hdi='higher').values,
                alpha=0.2, label='95% HDI'
            )
            axes[ax_idx].set_title('수준 컴포넌트')
            axes[ax_idx].legend()
            axes[ax_idx].grid(True)
            ax_idx += 1
        
        # 추세 컴포넌트 플롯
        if self.trend:
            trend_mean = self.trace.posterior.trend.mean(dim=["chain", "draw"]).values
            trend_hdi = az.hdi(self.trace.posterior.trend)
            
            axes[ax_idx].plot(trend_mean, label='추세 평균', color='orange')
            axes[ax_idx].fill_between(
                np.arange(len(trend_mean)),
                trend_hdi.sel(hdi='lower').values,
                trend_hdi.sel(hdi='higher').values,
                alpha=0.2, color='orange', label='95% HDI'
            )
            axes[ax_idx].set_title('추세 컴포넌트')
            axes[ax_idx].legend()
            axes[ax_idx].grid(True)
            ax_idx += 1
        
        # 계절성 컴포넌트 플롯
        if self.seasonality:
            seasonal_mean = self.trace.posterior.seasonal.mean(dim=["chain", "draw"]).values
            seasonal_hdi = az.hdi(self.trace.posterior.seasonal)
            
            axes[ax_idx].plot(seasonal_mean, label='계절성 평균', color='green')
            axes[ax_idx].fill_between(
                np.arange(len(seasonal_mean)),
                seasonal_hdi.sel(hdi='lower').values,
                seasonal_hdi.sel(hdi='higher').values,
                alpha=0.2, color='green', label='95% HDI'
            )
            axes[ax_idx].set_title('계절성 컴포넌트')
            axes[ax_idx].legend()
            axes[ax_idx].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_forecast(self, original_data: Union[pd.Series, np.ndarray], 
                     forecast: np.ndarray, 
                     lower: np.ndarray, 
                     upper: np.ndarray,
                     title: str = "구조적 시계열 모델 예측") -> plt.Figure:
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