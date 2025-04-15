import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from typing import List, Dict, Union, Tuple, Optional, Any
import logging
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class BayesianModelVisualizer:
    """
    베이지안 시계열 모델의 시각화를 위한 클래스
    
    다양한 시각화 기능을 제공합니다:
    - 예측 결과 시각화
    - 불확실성 구간 시각화
    - 사후 분포 시각화
    - 다중 모델 비교 시각화
    - 예측 성능 평가 시각화
    - 온라인 학습 성능 추적 시각화
    """
    
    def __init__(self, 
                 fig_size: Tuple[int, int] = (12, 6),
                 style: str = "whitegrid",
                 palette: str = "muted",
                 context: str = "notebook",
                 save_dir: str = "./plots"):
        """
        시각화 클래스 초기화
        
        Args:
            fig_size: 그래프 크기 (width, height)
            style: seaborn 스타일 ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
            palette: seaborn 색상 팔레트
            context: seaborn 컨텍스트 ('paper', 'notebook', 'talk', 'poster')
            save_dir: 그래프 저장 디렉토리
        """
        self.fig_size = fig_size
        self.style = style
        self.palette = palette
        self.context = context
        self.save_dir = save_dir
        
        # seaborn 설정
        sns.set_theme(style=style, palette=palette, context=context)
        
        # 저장 디렉토리 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"시각화 저장 디렉토리 생성: {save_dir}")
    
    def plot_forecast(self, 
                    original_data: Union[pd.Series, np.ndarray],
                    forecast: np.ndarray,
                    lower: np.ndarray,
                    upper: np.ndarray,
                    title: str = "베이지안 시계열 예측",
                    x_label: str = "시간",
                    y_label: str = "값",
                    dates: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
                    future_dates: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
                    show_uncertainty: bool = True,
                    confidence_level: str = "95%",
                    show_markers: bool = False,
                    highlight_last_point: bool = True,
                    model_name: Optional[str] = None,
                    save_path: Optional[str] = None) -> Figure:
        """
        예측 결과 시각화
        
        Args:
            original_data: 원본 시계열 데이터
            forecast: 예측값
            lower: 하한 신뢰구간
            upper: 상한 신뢰구간
            title: 그래프 제목
            x_label: x축 레이블
            y_label: y축 레이블
            dates: 날짜 데이터 (시계열 인덱스)
            future_dates: 예측 기간 날짜 데이터
            show_uncertainty: 불확실성 구간 표시 여부
            confidence_level: 신뢰구간 레벨 표시
            show_markers: 데이터 포인트에 마커 표시 여부
            highlight_last_point: 마지막 학습 데이터 포인트 강조 표시 여부
            model_name: 모델 이름 (그래프에 표시)
            save_path: 저장 경로 (None이면 저장하지 않음)
            
        Returns:
            그래프 객체
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # 데이터 준비
        if isinstance(original_data, pd.Series):
            y_data = original_data.values
            if dates is None and isinstance(original_data.index, pd.DatetimeIndex):
                dates = original_data.index
        else:
            y_data = original_data
        
        # x축 데이터 준비
        if dates is not None:
            x_data = dates
            if future_dates is None:
                # 날짜 추론
                if isinstance(dates, pd.DatetimeIndex):
                    freq = pd.infer_freq(dates)
                    if freq is None:
                        freq = 'D'  # 기본값
                    future_dates = pd.date_range(
                        start=dates[-1] + pd.Timedelta(days=1),
                        periods=len(forecast),
                        freq=freq
                    )
                else:
                    # 날짜 리스트인 경우
                    delta = dates[1] - dates[0]  # 첫 두 날짜 사이의 간격
                    future_dates = [dates[-1] + (i+1)*delta for i in range(len(forecast))]
            
            x_forecast = future_dates
        else:
            # 숫자 인덱스 사용
            x_data = np.arange(len(y_data))
            x_forecast = np.arange(len(y_data), len(y_data) + len(forecast))
        
        # 원본 데이터 플롯
        marker = 'o' if show_markers else None
        markersize = 4 if show_markers else None
        ax.plot(x_data, y_data, label='실제 데이터', color='#3366CC', marker=marker, markersize=markersize)
        
        # 예측 데이터 플롯
        ax.plot(x_forecast, forecast, label='예측', color='#DC3912', marker=marker, markersize=markersize)
        
        # 불확실성 구간 표시
        if show_uncertainty:
            ax.fill_between(x_forecast, lower, upper, alpha=0.2, color='#DC3912', 
                           label=f'{confidence_level} 신뢰구간')
        
        # 마지막 학습 데이터 포인트 강조
        if highlight_last_point:
            ax.scatter([x_data[-1]], [y_data[-1]], color='#3366CC', s=80, zorder=5, 
                      edgecolor='black', linewidth=1.5, label='마지막 학습 데이터')
        
        # 그래프 설정
        if model_name:
            title = f"{title} - {model_name}"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        # 날짜 형식 설정
        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        # 그리드 및 범례
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8, fontsize=10)
        
        plt.tight_layout()
        
        # 그래프 저장
        if save_path is not None:
            save_file = save_path if save_path.endswith(('.png', '.jpg', '.svg', '.pdf')) else f"{save_path}.png"
            full_path = os.path.join(self.save_dir, save_file)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"그래프 저장 완료: {full_path}")
        
        return fig
    
    def plot_multi_model_comparison(self,
                                  original_data: Union[pd.Series, np.ndarray],
                                  model_forecasts: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                  title: str = "모델 예측 비교",
                                  x_label: str = "시간",
                                  y_label: str = "값",
                                  dates: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
                                  future_dates: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
                                  show_uncertainty: bool = False,
                                  metrics: Optional[Dict[str, Dict[str, float]]] = None,
                                  save_path: Optional[str] = None) -> Figure:
        """
        여러 모델의 예측 결과 비교 시각화
        
        Args:
            original_data: 원본 시계열 데이터
            model_forecasts: 모델별 예측 결과 {모델명: (예측값, 하한, 상한)}
            title: 그래프 제목
            x_label: x축 레이블
            y_label: y축 레이블
            dates: 날짜 데이터 (시계열 인덱스)
            future_dates: 예측 기간 날짜 데이터
            show_uncertainty: 불확실성 구간 표시 여부
            metrics: 모델별 성능 지표 {'모델명': {'rmse': 값, 'mae': 값, ...}}
            save_path: 저장 경로 (None이면 저장하지 않음)
            
        Returns:
            그래프 객체
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # 데이터 준비
        if isinstance(original_data, pd.Series):
            y_data = original_data.values
            if dates is None and isinstance(original_data.index, pd.DatetimeIndex):
                dates = original_data.index
        else:
            y_data = original_data
        
        # x축 데이터 준비
        if dates is not None:
            x_data = dates
            if future_dates is None:
                # 예측 기간 길이 확인 (모든 모델이 같은 기간을 예측한다고 가정)
                n_forecast = len(next(iter(model_forecasts.values()))[0])
                
                # 날짜 추론
                if isinstance(dates, pd.DatetimeIndex):
                    freq = pd.infer_freq(dates)
                    if freq is None:
                        freq = 'D'  # 기본값
                    future_dates = pd.date_range(
                        start=dates[-1] + pd.Timedelta(days=1),
                        periods=n_forecast,
                        freq=freq
                    )
                else:
                    # 날짜 리스트인 경우
                    delta = dates[1] - dates[0]  # 첫 두 날짜 사이의 간격
                    future_dates = [dates[-1] + (i+1)*delta for i in range(n_forecast)]
            
            x_forecast = future_dates
        else:
            # 숫자 인덱스 사용
            x_data = np.arange(len(y_data))
            n_forecast = len(next(iter(model_forecasts.values()))[0])
            x_forecast = np.arange(len(y_data), len(y_data) + n_forecast)
        
        # 원본 데이터 플롯
        ax.plot(x_data, y_data, label='실제 데이터', color='black', linewidth=2)
        
        # 색상 순환
        colors = plt.cm.tab10.colors
        
        # 각 모델별 예측 플롯
        for i, (model_name, (forecast, lower, upper)) in enumerate(model_forecasts.items()):
            color = colors[i % len(colors)]
            
            # 예측 플롯
            label = model_name
            if metrics and model_name in metrics:
                # 지표를 레이블에 추가
                metric_str = ", ".join([f"{k.upper()}: {v:.4f}" for k, v in metrics[model_name].items() 
                                      if k in ['rmse', 'mae', 'mape']])
                label = f"{model_name} ({metric_str})"
            
            ax.plot(x_forecast, forecast, label=label, color=color, linewidth=1.5)
            
            # 불확실성 표시
            if show_uncertainty:
                ax.fill_between(x_forecast, lower, upper, alpha=0.15, color=color)
        
        # 그래프 설정
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        # 날짜 형식 설정
        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        # 그리드 및 범례
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8, fontsize=9)
        
        plt.tight_layout()
        
        # 그래프 저장
        if save_path is not None:
            save_file = save_path if save_path.endswith(('.png', '.jpg', '.svg', '.pdf')) else f"{save_path}.png"
            full_path = os.path.join(self.save_dir, save_file)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"그래프 저장 완료: {full_path}")
        
        return fig
    
    def plot_trace(self, trace, var_names=None, title="사후 분포 트레이스 플롯", 
                 figsize=(12, 10), save_path=None) -> Figure:
        """
        MCMC 샘플링 트레이스 플롯
        
        Args:
            trace: PyMC3 트레이스 객체
            var_names: 플롯할 변수 이름 목록
            title: 그래프 제목
            figsize: 그래프 크기
            save_path: 저장 경로 (None이면 저장하지 않음)
            
        Returns:
            그래프 객체
        """
        fig = plt.figure(figsize=figsize)
        
        # arviz 라이브러리를 사용한 트레이스 플롯
        ax = az.plot_trace(trace, var_names=var_names)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 그래프 저장
        if save_path is not None:
            save_file = save_path if save_path.endswith(('.png', '.jpg', '.svg', '.pdf')) else f"{save_path}.png"
            full_path = os.path.join(self.save_dir, save_file)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"트레이스 플롯 저장 완료: {full_path}")
        
        return fig
    
    def plot_posterior(self, trace, var_names=None, title="사후 분포", 
                     figsize=(12, 10), save_path=None) -> Figure:
        """
        사후 분포 플롯
        
        Args:
            trace: PyMC3 트레이스 객체
            var_names: 플롯할 변수 이름 목록
            title: 그래프 제목
            figsize: 그래프 크기
            save_path: 저장 경로 (None이면 저장하지 않음)
            
        Returns:
            그래프 객체
        """
        fig = plt.figure(figsize=figsize)
        
        # arviz 라이브러리를 사용한 사후 분포 플롯
        ax = az.plot_posterior(trace, var_names=var_names)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 그래프 저장
        if save_path is not None:
            save_file = save_path if save_path.endswith(('.png', '.jpg', '.svg', '.pdf')) else f"{save_path}.png"
            full_path = os.path.join(self.save_dir, save_file)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"사후 분포 플롯 저장 완료: {full_path}")
        
        return fig
    
    def plot_performance_metrics(self, 
                               metrics: Dict[str, Dict[str, float]],
                               title: str = "모델 성능 비교",
                               figsize: Tuple[int, int] = (12, 8),
                               primary_metric: str = "rmse",
                               save_path: Optional[str] = None) -> Figure:
        """
        모델 성능 지표 비교 플롯
        
        Args:
            metrics: 모델별 성능 지표 {'모델명': {'rmse': 값, 'mae': 값, ...}}
            title: 그래프 제목
            figsize: 그래프 크기
            primary_metric: 정렬 기준이 되는 주요 지표
            save_path: 저장 경로 (None이면 저장하지 않음)
            
        Returns:
            그래프 객체
        """
        # 주요 지표 추출 및 정렬
        if primary_metric not in next(iter(metrics.values())):
            logger.warning(f"주요 지표 {primary_metric}을(를) 찾을 수 없어 첫 번째 지표를 사용합니다.")
            primary_metric = next(iter(next(iter(metrics.values())).keys()))
        
        # 모델 이름과 주요 지표로 정렬
        sorted_models = sorted(metrics.items(), key=lambda x: x[1].get(primary_metric, float('inf')))
        model_names = [model[0] for model in sorted_models]
        
        # 모든 지표 목록 (중복 없이)
        all_metrics = set()
        for model_metrics in metrics.values():
            all_metrics.update(model_metrics.keys())
        
        # 시각화에 사용할 지표 필터링 (필요시)
        vis_metrics = [m for m in all_metrics if m in ['rmse', 'mae', 'mape', 'r2']]
        if not vis_metrics:
            vis_metrics = list(all_metrics)
        
        # 그래프 생성
        fig, axes = plt.subplots(len(vis_metrics), 1, figsize=figsize, sharex=True)
        if len(vis_metrics) == 1:
            axes = [axes]  # 단일 지표인 경우 리스트로 변환
        
        # 색상 맵
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(len(model_names))]
        
        for i, metric in enumerate(vis_metrics):
            ax = axes[i]
            
            # 지표 값 추출 (없는 경우 NaN)
            values = [model_metric.get(metric, np.nan) for _, model_metric in sorted_models]
            
            # 바 플롯
            bars = ax.bar(model_names, values, color=colors, alpha=0.7)
            
            # 값 표시
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(values),
                           f'{value:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 축 레이블 및 그리드
            ax.set_ylabel(metric.upper())
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 최적 모델 강조
            if metric == primary_metric:
                idx_min = values.index(min([v for v in values if not np.isnan(v)]))
                bars[idx_min].set_color('green')
                bars[idx_min].set_alpha(0.9)
        
        # x축 레이블 회전
        plt.xticks(rotation=45, ha='right')
        
        # 제목 및 레이아웃
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 그래프 저장
        if save_path is not None:
            save_file = save_path if save_path.endswith(('.png', '.jpg', '.svg', '.pdf')) else f"{save_path}.png"
            full_path = os.path.join(self.save_dir, save_file)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"성능 지표 플롯 저장 완료: {full_path}")
        
        return fig
    
    def plot_online_learning_performance(self,
                                       performance_history: List[Dict[str, Any]],
                                       metric: str = "rmse",
                                       title: str = "온라인 학습 성능 추적",
                                       figsize: Tuple[int, int] = (12, 6),
                                       show_update_points: bool = True,
                                       save_path: Optional[str] = None) -> Figure:
        """
        온라인 학습 성능 추적 플롯
        
        Args:
            performance_history: 성능 기록 목록
            metric: 플롯할 지표
            title: 그래프 제목
            figsize: 그래프 크기
            show_update_points: 모델 업데이트 시점 표시 여부
            save_path: 저장 경로 (None이면 저장하지 않음)
            
        Returns:
            그래프 객체
        """
        if not performance_history:
            logger.warning("성능 기록이 비어 있습니다.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "데이터 없음", ha='center', va='center', fontsize=14)
            return fig
        
        # 데이터 추출
        timestamps = [entry['timestamp'] for entry in performance_history]
        metrics = [entry['metrics'].get(metric, np.nan) for entry in performance_history]
        updates = [entry.get('is_update', False) for entry in performance_history]
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=figsize)
        
        # 성능 지표 플롯
        ax.plot(timestamps, metrics, label=metric.upper(), marker='o', markersize=3)
        
        # 모델 업데이트 시점 표시
        if show_update_points:
            update_timestamps = [ts for ts, upd in zip(timestamps, updates) if upd]
            update_metrics = [met for met, upd in zip(metrics, updates) if upd]
            
            if update_timestamps:
                ax.scatter(update_timestamps, update_metrics, color='red', s=80, zorder=5,
                         label='모델 업데이트', alpha=0.7)
                
                # 업데이트 시점에 수직선 추가
                for ts in update_timestamps:
                    ax.axvline(x=ts, color='red', linestyle='--', alpha=0.3)
        
        # 그래프 설정
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('시간')
        ax.set_ylabel(metric.upper())
        
        # 날짜 형식 설정
        if isinstance(timestamps[0], (datetime, np.datetime64, pd.Timestamp)):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
        
        # 그리드 및 범례
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # 그래프 저장
        if save_path is not None:
            save_file = save_path if save_path.endswith(('.png', '.jpg', '.svg', '.pdf')) else f"{save_path}.png"
            full_path = os.path.join(self.save_dir, save_file)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"온라인 학습 성능 플롯 저장 완료: {full_path}")
        
        return fig 