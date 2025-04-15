import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Dict, List, Tuple, Optional, Any, Callable
import logging
import time
from datetime import datetime
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .model_factory import BayesianModelFactory
from .ensemble_model import BayesianEnsembleModel

logger = logging.getLogger(__name__)

class OnlineBayesianLearner:
    """
    온라인 학습을 지원하는 베이지안 시계열 모델 래퍼 클래스
    
    새로운 데이터가 들어올 때마다 모델을 업데이트하는 온라인 학습 방법을 제공합니다.
    """
    
    def __init__(self, 
                 model_type: str = "ar", 
                 model_params: Optional[Dict[str, Any]] = None,
                 window_size: int = 180,
                 update_freq: int = 7,
                 save_dir: str = "./models",
                 auto_save: bool = True,
                 ensemble_config: Optional[List[Dict[str, Any]]] = None,
                 ensemble_method: str = "weighted",
                 eval_metric: str = "rmse"):
        """
        온라인 베이지안 학습기 초기화
        
        Args:
            model_type: 모델 유형 ('ar', 'gp', 'structural', 'ensemble')
            model_params: 모델 파라미터
            window_size: 학습에 사용할 최대 데이터 포인트 수
            update_freq: 모델 업데이트 주기 (데이터 포인트 단위)
            save_dir: 모델 저장 디렉터리
            auto_save: 자동 저장 여부
            ensemble_config: 앙상블 모델 구성 (model_type이 'ensemble'인 경우 필요)
            ensemble_method: 앙상블 방법 (model_type이 'ensemble'인 경우 사용)
            eval_metric: 평가 지표 ('rmse', 'mae', 'mape')
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.window_size = window_size
        self.update_freq = update_freq
        self.save_dir = save_dir
        self.auto_save = auto_save
        self.ensemble_config = ensemble_config
        self.ensemble_method = ensemble_method
        self.eval_metric = eval_metric
        
        # 모델 및 데이터 초기화
        self.model = None
        self.data_buffer = []
        self.last_update_time = None
        self.last_update_size = 0
        self.update_count = 0
        self.performance_history = []
        
        # 예측 결과 저장
        self.last_forecast = None
        self.last_lower = None
        self.last_upper = None
        
        # 평가 지표 함수 설정
        if eval_metric == "rmse":
            self.eval_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
        elif eval_metric == "mae":
            self.eval_func = lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)
        elif eval_metric == "mape":
            self.eval_func = lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        else:
            logger.warning(f"지원하지 않는 평가 지표: {eval_metric}, 기본값 RMSE로 설정합니다.")
            self.eval_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
        
        # 저장 디렉터리 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"모델 저장 디렉터리 생성: {save_dir}")
        
        # 모델 초기화
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        모델 초기화
        """
        try:
            if self.model_type == "ensemble":
                if not self.ensemble_config:
                    raise ValueError("ensemble_config가 필요합니다.")
                
                self.model = BayesianEnsembleModel(
                    models_config=self.ensemble_config,
                    ensemble_method=self.ensemble_method
                )
                logger.info(f"앙상블 모델 초기화 완료: {len(self.ensemble_config)}개 모델, 방식: {self.ensemble_method}")
            else:
                self.model = BayesianModelFactory.get_model(
                    model_type=self.model_type,
                    **self.model_params
                )
                logger.info(f"{self.model_type} 모델 초기화 완료")
        except Exception as e:
            logger.error(f"모델 초기화 실패: {str(e)}")
            raise
    
    def add_data(self, new_data: Union[float, List[float], np.ndarray, pd.Series],
               timestamps: Optional[Union[List[datetime], pd.DatetimeIndex]] = None,
               update_model: bool = True) -> bool:
        """
        새 데이터 추가 및 필요시 모델 업데이트
        
        Args:
            new_data: 새로운 데이터 포인트(들)
            timestamps: 타임스탬프(들) (선택 사항)
            update_model: 모델 업데이트 여부
            
        Returns:
            업데이트 성공 여부
        """
        # 단일 값 -> 리스트 변환
        if isinstance(new_data, (float, int)):
            new_data = [new_data]
            if timestamps is not None and not isinstance(timestamps, list):
                timestamps = [timestamps]
        
        # pandas Series -> 값 추출
        if isinstance(new_data, pd.Series):
            if timestamps is None:
                timestamps = new_data.index
            new_data = new_data.values
        
        # 타임스탬프 없으면 현재 시간 사용
        if timestamps is None:
            now = datetime.now()
            timestamps = [now + pd.Timedelta(seconds=i) for i in range(len(new_data))]
        
        # 데이터 버퍼에 추가
        for i, value in enumerate(new_data):
            if i < len(timestamps):
                self.data_buffer.append((timestamps[i], value))
            else:
                self.data_buffer.append((datetime.now(), value))
        
        # 슬라이딩 윈도우 유지 (최대 크기로 제한)
        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size:]
        
        # 업데이트 주기 확인
        data_size = len(self.data_buffer)
        points_since_update = data_size - self.last_update_size
        
        if update_model and points_since_update >= self.update_freq:
            logger.info(f"모델 업데이트 조건 충족: {points_since_update}개 새 데이터 포인트")
            return self.update()
        
        return False
    
    def update(self, force: bool = False) -> bool:
        """
        모델 업데이트
        
        Args:
            force: 강제 업데이트 여부
            
        Returns:
            업데이트 성공 여부
        """
        data_size = len(self.data_buffer)
        
        # 데이터가 충분한지 확인
        if data_size < 30 and not force:  # 최소 30개 데이터 포인트 필요
            logger.warning(f"모델 업데이트를 위한 데이터가 부족합니다: {data_size}/30")
            return False
        
        # 새 데이터가 있는지 확인
        points_since_update = data_size - self.last_update_size
        if points_since_update == 0 and not force:
            logger.info("새 데이터가 없어 모델 업데이트를 건너뜁니다.")
            return False
        
        try:
            # 데이터 추출
            timestamps, values = zip(*self.data_buffer)
            
            # pandas Series로 변환
            data_series = pd.Series(values, index=timestamps)
            
            # 학습/검증 분할
            train_size = int(data_size * 0.8)
            train_data = data_series[:train_size]
            val_data = data_series[train_size:]
            
            # MCMC 파라미터 (속도를 위해 간소화)
            sampling_params = {
                'draws': 300,
                'tune': 300,
                'chains': 2,
                'target_accept': 0.9
            }
            
            # 모델 학습
            logger.info(f"모델 학습 중: {data_size}개 데이터 포인트 ({points_since_update}개 새 데이터)")
            start_time = time.time()
            
            self.model.fit(train_data, sampling_params=sampling_params)
            
            train_time = time.time() - start_time
            logger.info(f"모델 학습 완료: {train_time:.2f}초 소요")
            
            # 검증 데이터로 평가
            if len(val_data) > 0:
                val_metrics = self._evaluate_model(val_data)
                logger.info(f"검증 성능: {self.eval_metric.upper()} = {val_metrics[self.eval_metric]:.4f}")
                
                # 성능 기록 저장
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'data_size': data_size,
                    'train_size': len(train_data),
                    'val_size': len(val_data),
                    'train_time': train_time,
                    'metrics': val_metrics
                })
            
            # 모델 자동 저장
            if self.auto_save:
                self.save_model()
            
            # 업데이트 상태 갱신
            self.last_update_time = datetime.now()
            self.last_update_size = data_size
            self.update_count += 1
            
            return True
            
        except Exception as e:
            logger.error(f"모델 업데이트 실패: {str(e)}")
            return False
    
    def predict(self, n_forecast: int = 7, return_conf_int: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        미래 예측 수행
        
        Args:
            n_forecast: 예측할 미래 시점 수
            return_conf_int: 신뢰 구간 반환 여부
            
        Returns:
            예측값 (또는 예측값, 하한, 상한)
        """
        if self.model is None or not hasattr(self.model, 'predict'):
            raise ValueError("모델이 초기화되지 않았거나 예측을 지원하지 않습니다.")
        
        try:
            forecast, lower, upper = self.model.predict(n_forecast=n_forecast)
            
            # 예측 결과 저장
            self.last_forecast = forecast
            self.last_lower = lower
            self.last_upper = upper
            
            if return_conf_int:
                return forecast, lower, upper
            else:
                return forecast
                
        except Exception as e:
            logger.error(f"예측 실패: {str(e)}")
            raise
    
    def _evaluate_model(self, test_data: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        테스트 데이터로 모델 평가
        
        Args:
            test_data: 테스트 데이터
            
        Returns:
            평가 지표 딕셔너리
        """
        # 테스트 데이터 길이만큼 예측
        n_test = len(test_data)
        forecast, _, _ = self.model.predict(n_forecast=n_test)
        
        # 테스트 데이터가 Series인 경우 값 추출
        if isinstance(test_data, pd.Series):
            test_values = test_data.values
        else:
            test_values = test_data
        
        # 평가 지표 계산
        mse = mean_squared_error(test_values, forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_values, forecast)
        
        try:
            mape = np.mean(np.abs((test_values - forecast) / test_values)) * 100
        except:
            mape = np.nan
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def plot_forecast(self, n_forecast: int = 30, show_history: bool = True,
                    title: Optional[str] = None, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        예측 결과 시각화
        
        Args:
            n_forecast: 예측할 미래 시점 수
            show_history: 과거 데이터 표시 여부
            title: 그래프 제목
            figsize: 그래프 크기
            
        Returns:
            그래프 객체
        """
        if not self.data_buffer:
            raise ValueError("데이터가 없습니다.")
        
        # 예측 수행
        if (self.last_forecast is None or len(self.last_forecast) != n_forecast):
            forecast, lower, upper = self.predict(n_forecast=n_forecast)
        else:
            forecast, lower, upper = self.last_forecast, self.last_lower, self.last_upper
        
        # 과거 데이터 추출
        timestamps, values = zip(*self.data_buffer)
        history = pd.Series(values, index=timestamps)
        
        # 미래 날짜 생성
        last_date = timestamps[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(pd.DatetimeIndex(timestamps[-10:]))
            if freq is None:
                freq = 'D'  # 기본값 일단위
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=n_forecast, 
                                       freq=freq)
        else:
            future_dates = pd.date_range(start=pd.Timestamp.now(), 
                                       periods=n_forecast, 
                                       freq='D')
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 과거 데이터 표시
        if show_history:
            ax.plot(history.index, history.values, label='과거 데이터', color='blue')
        
        # 예측 표시
        ax.plot(future_dates, forecast, label='예측', color='red')
        ax.fill_between(future_dates, lower, upper, alpha=0.2, color='red', label='95% 신뢰 구간')
        
        # 제목 및 레이블
        if title is None:
            title = f"베이지안 시계열 예측 ({self.model_type} 모델)"
        ax.set_title(title)
        ax.set_xlabel('날짜')
        ax.set_ylabel('값')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_performance(self, metric: Optional[str] = None, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        모델 성능 추이 시각화
        
        Args:
            metric: 표시할 평가 지표 (None이면 기본 평가 지표 사용)
            figsize: 그래프 크기
            
        Returns:
            그래프 객체
        """
        if not self.performance_history:
            raise ValueError("성능 기록이 없습니다.")
        
        if metric is None:
            metric = self.eval_metric
        
        # 데이터 추출
        timestamps = [entry['timestamp'] for entry in self.performance_history]
        metrics = [entry['metrics'][metric] for entry in self.performance_history]
        data_sizes = [entry['data_size'] for entry in self.performance_history]
        
        # 그래프 생성
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # 메트릭 그래프
        color = 'tab:red'
        ax1.set_xlabel('날짜')
        ax1.set_ylabel(f'{metric.upper()}', color=color)
        ax1.plot(timestamps, metrics, marker='o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 데이터 크기 그래프
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('데이터 크기', color=color)
        ax2.plot(timestamps, data_sizes, marker='s', linestyle='--', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f"모델 성능 추이 ({metric.upper()})")
        fig.tight_layout()
        return fig
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        모델 저장
        
        Args:
            filename: 저장 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_model_{timestamp}.pkl"
        
        file_path = os.path.join(self.save_dir, filename)
        
        try:
            # 모델 데이터 구성
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'model_params': self.model_params,
                'data_buffer': self.data_buffer,
                'last_update_time': self.last_update_time,
                'last_update_size': self.last_update_size,
                'update_count': self.update_count,
                'performance_history': self.performance_history,
                'window_size': self.window_size,
                'update_freq': self.update_freq
            }
            
            # 모델 저장
            joblib.dump(model_data, file_path)
            logger.info(f"모델 저장 완료: {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"모델 저장 실패: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, file_path: str) -> 'OnlineBayesianLearner':
        """
        저장된 모델 로드
        
        Args:
            file_path: 모델 파일 경로
            
        Returns:
            OnlineBayesianLearner 인스턴스
        """
        try:
            # 모델 데이터 로드
            model_data = joblib.load(file_path)
            
            # 인스턴스 생성
            learner = cls(
                model_type=model_data['model_type'],
                model_params=model_data['model_params'],
                window_size=model_data['window_size'],
                update_freq=model_data['update_freq']
            )
            
            # 데이터 복원
            learner.model = model_data['model']
            learner.data_buffer = model_data['data_buffer']
            learner.last_update_time = model_data['last_update_time']
            learner.last_update_size = model_data['last_update_size']
            learner.update_count = model_data['update_count']
            learner.performance_history = model_data['performance_history']
            
            logger.info(f"모델 로드 완료: {file_path}")
            
            return learner
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            raise 