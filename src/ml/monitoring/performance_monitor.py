import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from scipy import stats
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

@dataclass
class MetricsSnapshot:
    """모델 성능 지표 스냅샷"""
    model_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    data_size: int
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'data_size': self.data_size,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricsSnapshot':
        """딕셔너리에서 객체 생성"""
        return cls(
            model_id=data['model_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metrics=data['metrics'],
            data_size=data['data_size'],
            training_time=data.get('training_time'),
            prediction_time=data.get('prediction_time')
        )

class MLPerformanceMonitor:
    """머신러닝 모델 성능 모니터링 시스템
    
    모델 성능 지표를 추적하고 시각화하는 기능 제공:
    - 시간에 따른 성능 지표 추적
    - 성능 저하 감지 및 알림
    - 데이터 드리프트 감지
    - 성능 시각화 대시보드
    """
    
    def __init__(self, save_dir: str = 'logs/ml_performance',
                config: Dict[str, Any] = None):
        """
        Args:
            save_dir: 로그 및 시각화 파일 저장 경로
            config: 설정 파라미터
        """
        self.save_dir = save_dir
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'snapshots'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
        
        # 기본 설정
        self.config = {
            'alert_threshold': 0.1,  # 성능 저하 알림 임계값 (10%)
            'window_size': 5,        # 이동 평균 윈도우 크기
            'log_to_file': True,     # 파일에 로그 저장 여부
            'plot_format': 'png',    # 그래프 저장 형식
            'drift_detection_method': 'ks_test',  # 드리프트 감지 방법
            'drift_threshold': 0.05,  # 드리프트 감지 임계값 (p-value)
        }
        
        # 사용자 설정 병합
        if config:
            self.config.update(config)
        
        # 성능 지표 기록
        self.metrics_history = {}
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
        # 저장된 데이터 로드
        self._load_history()
    
    def _load_history(self) -> None:
        """저장된 성능 기록 로드"""
        snapshots_dir = os.path.join(self.save_dir, 'snapshots')
        if not os.path.exists(snapshots_dir):
            return
        
        # 각 모델에 대한 스냅샷 파일 로드
        for filename in os.listdir(snapshots_dir):
            if filename.endswith('.json'):
                try:
                    file_path = os.path.join(snapshots_dir, filename)
                    with open(file_path, 'r') as f:
                        snapshots = json.load(f)
                    
                    model_id = filename.split('_')[0]
                    self.metrics_history[model_id] = []
                    
                    for snapshot_data in snapshots:
                        snapshot = MetricsSnapshot.from_dict(snapshot_data)
                        self.metrics_history[model_id].append(snapshot)
                    
                    # 시간순 정렬
                    self.metrics_history[model_id].sort(key=lambda x: x.timestamp)
                    
                    self.logger.info(f"모델 {model_id}의 성능 기록 {len(self.metrics_history[model_id])}개 로드")
                    
                except Exception as e:
                    self.logger.error(f"스냅샷 파일 {filename} 로드 중 오류: {str(e)}")
    
    def log_metrics(self, model_id: str, metrics: Dict[str, float], 
                   data_size: int, timestamp: Optional[datetime] = None,
                   training_time: Optional[float] = None,
                   prediction_time: Optional[float] = None) -> None:
        """모델 성능 지표 기록
        
        Args:
            model_id: 모델 식별자
            metrics: 성능 지표 (예: {'rmse': 0.123, 'mae': 0.089})
            data_size: 평가 데이터 크기
            timestamp: 타임스탬프 (None이면 현재 시간)
            training_time: 훈련 시간 (초)
            prediction_time: 예측 시간 (초)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 새 스냅샷 생성
        snapshot = MetricsSnapshot(
            model_id=model_id,
            timestamp=timestamp,
            metrics=metrics,
            data_size=data_size,
            training_time=training_time,
            prediction_time=prediction_time
        )
        
        # 모델 ID가 없으면 새로 생성
        if model_id not in self.metrics_history:
            self.metrics_history[model_id] = []
        
        # 스냅샷 추가
        self.metrics_history[model_id].append(snapshot)
        
        # 성능 저하 감지
        alerts = self._check_performance_degradation(model_id)
        if alerts:
            for metric, alert in alerts.items():
                self.logger.warning(
                    f"모델 {model_id}의 {metric} 성능 저하 감지: "
                    f"{alert['previous']:.6f} -> {alert['current']:.6f} "
                    f"({alert['change']:.2%} 변화)"
                )
        
        # 결과 저장
        self._save_metrics_snapshot(model_id)
    
    def _save_metrics_snapshot(self, model_id: str) -> None:
        """성능 지표 스냅샷 저장
        
        Args:
            model_id: 모델 식별자
        """
        if not self.config.get('log_to_file', True):
            return
        
        # 저장 파일 경로
        file_path = os.path.join(self.save_dir, 'snapshots', f"{model_id}_snapshots.json")
        
        # 스냅샷 데이터 직렬화
        snapshots_data = [s.to_dict() for s in self.metrics_history[model_id]]
        
        # JSON으로 저장
        with open(file_path, 'w') as f:
            json.dump(snapshots_data, f, indent=2)
    
    def _check_performance_degradation(self, model_id: str) -> Dict[str, Dict[str, float]]:
        """성능 저하 감지
        
        Args:
            model_id: 모델 식별자
            
        Returns:
            성능 저하 알림 딕셔너리 (지표별)
        """
        history = self.metrics_history.get(model_id)
        if not history or len(history) < 2:
            return {}
        
        # 가장 최근 스냅샷과 이전 스냅샷 비교
        current = history[-1]
        
        # 이동 평균 사용 (윈도우 크기 설정)
        window_size = min(self.config.get('window_size', 5), len(history) - 1)
        if window_size < 1:
            previous = history[-2]
            previous_metrics = previous.metrics
        else:
            # 마지막 스냅샷 제외한 윈도우 기간의 평균
            previous_snapshots = history[-(window_size+1):-1]
            
            # 각 지표별 이동 평균 계산
            previous_metrics = {}
            for metric in current.metrics.keys():
                values = [s.metrics.get(metric, 0) for s in previous_snapshots if metric in s.metrics]
                if values:
                    previous_metrics[metric] = sum(values) / len(values)
        
        # 알림 임계값
        threshold = self.config.get('alert_threshold', 0.1)
        
        # 성능 저하 감지
        alerts = {}
        for metric, value in current.metrics.items():
            if metric not in previous_metrics:
                continue
            
            previous_value = previous_metrics[metric]
            
            # 변화율 계산
            if previous_value != 0:
                change = (value - previous_value) / abs(previous_value)
            else:
                change = 0 if value == 0 else float('inf')
            
            # 지표 증가가 좋은지 나쁜지 결정
            is_improvement = metric.lower() in ['accuracy', 'r2', 'precision', 'recall', 'f1']
            
            # 저하 여부 확인
            if (is_improvement and change < -threshold) or (not is_improvement and change > threshold):
                alerts[metric] = {
                    'current': value,
                    'previous': previous_value,
                    'change': change
                }
        
        return alerts
    
    def get_model_metrics_trend(self, model_id: str, 
                               metric_name: str) -> pd.DataFrame:
        """시간에 따른 모델 성능 지표 추세 반환
        
        Args:
            model_id: 모델 식별자
            metric_name: 지표 이름
            
        Returns:
            시간-지표 데이터프레임
        """
        if model_id not in self.metrics_history:
            raise ValueError(f"모델 ID {model_id}에 대한 기록이 없습니다")
        
        history = self.metrics_history[model_id]
        
        # 타임스탬프와 지표 추출
        data = []
        for snapshot in history:
            if metric_name in snapshot.metrics:
                data.append({
                    'timestamp': snapshot.timestamp,
                    'value': snapshot.metrics[metric_name],
                    'data_size': snapshot.data_size
                })
        
        if not data:
            raise ValueError(f"지표 {metric_name}에 대한 기록이 없습니다")
        
        return pd.DataFrame(data)
    
    def plot_performance_trend(self, model_id: str, 
                             metric_name: str = 'rmse',
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             show_plot: bool = True,
                             save_path: Optional[str] = None) -> str:
        """모델 성능 지표 추세 시각화
        
        Args:
            model_id: 모델 식별자
            metric_name: 지표 이름
            start_date: 시작 날짜
            end_date: 종료 날짜
            show_plot: 그래프 표시 여부
            save_path: 저장 경로 (None이면 자동 생성)
            
        Returns:
            그래프 저장 경로 또는 빈 문자열
        """
        try:
            # 지표 추세 데이터 가져오기
            trend_data = self.get_model_metrics_trend(model_id, metric_name)
            
            # 날짜 필터링
            if start_date:
                trend_data = trend_data[trend_data['timestamp'] >= start_date]
            if end_date:
                trend_data = trend_data[trend_data['timestamp'] <= end_date]
            
            if trend_data.empty:
                self.logger.warning(f"지정된 기간에 대한 데이터가 없습니다")
                return ""
            
            # 그래프 생성
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # 성능 지표 플롯
            color = 'tab:blue'
            ax1.set_xlabel('날짜')
            ax1.set_ylabel(metric_name.upper(), color=color)
            ax1.plot(trend_data['timestamp'], trend_data['value'], 
                    'o-', color=color, label=metric_name.upper())
            ax1.tick_params(axis='y', labelcolor=color)
            
            # 이동 평균 추가
            window = min(5, len(trend_data))
            if window > 1:
                trend_data['moving_avg'] = trend_data['value'].rolling(window=window).mean()
                ax1.plot(trend_data['timestamp'], trend_data['moving_avg'], 
                        '--', color='navy', alpha=0.7, 
                        label=f'{window}포인트 이동 평균')
            
            # 데이터 크기 플롯 (보조 y축)
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('데이터 크기', color=color)
            ax2.plot(trend_data['timestamp'], trend_data['data_size'], 
                    'o--', color=color, alpha=0.5, label='데이터 크기')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # 날짜 형식 설정
            date_format = mdates.DateFormatter('%Y-%m-%d')
            ax1.xaxis.set_major_formatter(date_format)
            fig.autofmt_xdate()  # 날짜 레이블 자동 포맷팅
            
            # 그리드 및 범례
            ax1.grid(True, linestyle='--', alpha=0.7)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title(f'모델 {model_id}의 {metric_name.upper()} 추세')
            plt.tight_layout()
            
            # 그래프 저장
            if save_path is None:
                format = self.config.get('plot_format', 'png')
                filename = f"{model_id}_{metric_name}_trend.{format}"
                save_path = os.path.join(self.save_dir, 'visualizations', filename)
            
            plt.savefig(save_path)
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"성능 추세 시각화 중 오류: {str(e)}")
            return ""
    
    def plot_metrics_comparison(self, models: List[str], 
                              metric_name: str = 'rmse',
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              show_plot: bool = True,
                              save_path: Optional[str] = None) -> str:
        """여러 모델의 성능 지표 비교 시각화
        
        Args:
            models: 모델 식별자 목록
            metric_name: 지표 이름
            start_date: 시작 날짜
            end_date: 종료 날짜
            show_plot: 그래프 표시 여부
            save_path: 저장 경로 (None이면 자동 생성)
            
        Returns:
            그래프 저장 경로 또는 빈 문자열
        """
        try:
            plt.figure(figsize=(12, 6))
            
            for model_id in models:
                if model_id not in self.metrics_history:
                    self.logger.warning(f"모델 ID {model_id}에 대한 기록이 없습니다")
                    continue
                
                # 지표 추세 데이터 가져오기
                try:
                    trend_data = self.get_model_metrics_trend(model_id, metric_name)
                    
                    # 날짜 필터링
                    if start_date:
                        trend_data = trend_data[trend_data['timestamp'] >= start_date]
                    if end_date:
                        trend_data = trend_data[trend_data['timestamp'] <= end_date]
                    
                    if not trend_data.empty:
                        plt.plot(trend_data['timestamp'], trend_data['value'], 
                                'o-', label=f'모델 {model_id}')
                except ValueError:
                    continue
            
            # 날짜 형식 설정
            date_format = mdates.DateFormatter('%Y-%m-%d')
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.gcf().autofmt_xdate()  # 날짜 레이블 자동 포맷팅
            
            plt.xlabel('날짜')
            plt.ylabel(metric_name.upper())
            plt.title(f'모델 간 {metric_name.upper()} 비교')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            # 그래프 저장
            if save_path is None:
                format = self.config.get('plot_format', 'png')
                model_ids = '_'.join(models)
                if len(model_ids) > 30:
                    model_ids = f"{len(models)}models"
                filename = f"{model_ids}_{metric_name}_comparison.{format}"
                save_path = os.path.join(self.save_dir, 'visualizations', filename)
            
            plt.savefig(save_path)
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"모델 비교 시각화 중 오류: {str(e)}")
            return ""
    
    def get_latest_metrics(self, model_id: str) -> Optional[Dict[str, float]]:
        """모델의 최신 성능 지표 반환
        
        Args:
            model_id: 모델 식별자
            
        Returns:
            최신 성능 지표 딕셔너리 또는 None
        """
        if model_id not in self.metrics_history or not self.metrics_history[model_id]:
            return None
        
        return self.metrics_history[model_id][-1].metrics
    
    def detect_data_drift(self, model_id: str, 
                        reference_data: pd.DataFrame,
                        current_data: pd.DataFrame,
                        features: List[str],
                        threshold: Optional[float] = None) -> Dict[str, Any]:
        """특성 분포 변화 감지
        
        Args:
            model_id: 모델 식별자
            reference_data: 기준 데이터
            current_data: 현재 데이터
            features: 드리프트 검사할 특성 목록
            threshold: 드리프트 감지 임계값 (None이면 설정에서 가져옴)
            
        Returns:
            드리프트 감지 결과
        """
        if threshold is None:
            threshold = self.config.get('drift_threshold', 0.05)
        
        drift_method = self.config.get('drift_detection_method', 'ks_test')
        
        # 결과 저장
        results = {
            'model_id': model_id,
            'timestamp': datetime.now(),
            'drift_detected': False,
            'drifted_features': [],
            'all_features': features,
            'method': drift_method,
            'threshold': threshold,
            'results': {}
        }
        
        # 각 특성에 대해 드리프트 검사
        for feature in features:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue
            
            ref_data = reference_data[feature].dropna()
            cur_data = current_data[feature].dropna()
            
            if len(ref_data) < 10 or len(cur_data) < 10:
                continue
            
            # 드리프트 검사 방법
            if drift_method == 'ks_test':
                # 콜모고로프-스미르노프 검정
                statistic, p_value = stats.ks_2samp(ref_data, cur_data)
                drift_detected = p_value < threshold
                
                results['results'][feature] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'drift_detected': drift_detected
                }
                
                if drift_detected:
                    results['drifted_features'].append(feature)
                    results['drift_detected'] = True
            
            elif drift_method == 'chi2_test':
                # 카이제곱 검정 (범주형 데이터용)
                try:
                    # 히스토그램으로 빈도수 계산
                    hist1, bin_edges = np.histogram(ref_data, bins=10)
                    hist2, _ = np.histogram(cur_data, bins=bin_edges)
                    
                    # 최소 빈도수 체크
                    if np.min(hist1) >= 5 and np.min(hist2) >= 5:
                        statistic, p_value = stats.chisquare(hist1, hist2)
                        drift_detected = p_value < threshold
                        
                        results['results'][feature] = {
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'drift_detected': drift_detected
                        }
                        
                        if drift_detected:
                            results['drifted_features'].append(feature)
                            results['drift_detected'] = True
                except Exception as e:
                    self.logger.warning(f"특성 {feature}의 카이제곱 검정 중 오류: {str(e)}")
        
        # 드리프트 결과 로깅
        self.logger.info(f"모델 {model_id} 데이터 드리프트 감지 결과: "
                       f"{len(results['drifted_features'])}/{len(features)} "
                       f"특성에서 드리프트 감지")
        
        if results['drift_detected']:
            for feature in results['drifted_features']:
                self.logger.warning(f"모델 {model_id}의 특성 '{feature}'에서 "
                                   f"드리프트 감지 (p-value: {results['results'][feature]['p_value']:.6f})")
        
        return results
    
    def plot_feature_drift(self, model_id: str,
                         reference_data: pd.DataFrame,
                         current_data: pd.DataFrame,
                         feature: str,
                         show_plot: bool = True,
                         save_path: Optional[str] = None) -> str:
        """특성 분포 드리프트 시각화
        
        Args:
            model_id: 모델 식별자
            reference_data: 기준 데이터
            current_data: 현재 데이터
            feature: 특성 이름
            show_plot: 그래프 표시 여부
            save_path: 저장 경로 (None이면 자동 생성)
            
        Returns:
            그래프 저장 경로 또는 빈 문자열
        """
        try:
            if feature not in reference_data.columns or feature not in current_data.columns:
                raise ValueError(f"특성 {feature}가 데이터에 없습니다")
            
            ref_data = reference_data[feature].dropna()
            cur_data = current_data[feature].dropna()
            
            if len(ref_data) < 10 or len(cur_data) < 10:
                raise ValueError(f"특성 {feature}의 데이터가 부족합니다")
            
            # 그래프 설정
            plt.figure(figsize=(12, 6))
            
            # 히스토그램 및 KDE (커널 밀도 추정) 플롯
            sns.histplot(ref_data, color='blue', alpha=0.5, 
                       label='기준 데이터', kde=True)
            sns.histplot(cur_data, color='red', alpha=0.5, 
                       label='현재 데이터', kde=True)
            
            # 드리프트 검정
            statistic, p_value = stats.ks_2samp(ref_data, cur_data)
            drift_detected = p_value < self.config.get('drift_threshold', 0.05)
            
            drift_status = "감지됨" if drift_detected else "감지되지 않음"
            plt.title(f"특성 '{feature}'의 분포 드리프트 ({drift_status}, p-value: {p_value:.6f})")
            plt.xlabel(feature)
            plt.ylabel('빈도')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 그래프 저장
            if save_path is None:
                format = self.config.get('plot_format', 'png')
                filename = f"{model_id}_{feature}_drift.{format}"
                save_path = os.path.join(self.save_dir, 'visualizations', filename)
            
            plt.savefig(save_path)
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"특성 드리프트 시각화 중 오류: {str(e)}")
            return ""
    
    def generate_performance_report(self, model_id: str,
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """모델 성능 보고서 생성
        
        Args:
            model_id: 모델 식별자
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            성능 보고서 딕셔너리
        """
        if model_id not in self.metrics_history:
            raise ValueError(f"모델 ID {model_id}에 대한 기록이 없습니다")
        
        history = self.metrics_history[model_id]
        
        # 날짜 필터링
        filtered_history = history
        if start_date or end_date:
            filtered_history = []
            for snapshot in history:
                if start_date and snapshot.timestamp < start_date:
                    continue
                if end_date and snapshot.timestamp > end_date:
                    continue
                filtered_history.append(snapshot)
        
        if not filtered_history:
            raise ValueError("지정된 기간에 대한 데이터가 없습니다")
        
        # 최신 및 초기 지표
        latest = filtered_history[-1]
        first = filtered_history[0]
        
        # 각 지표별 통계
        metrics_stats = {}
        for metric in latest.metrics:
            values = [s.metrics.get(metric) for s in filtered_history if metric in s.metrics]
            values = [v for v in values if v is not None]
            
            if not values:
                continue
            
            metrics_stats[metric] = {
                'current': latest.metrics.get(metric),
                'first': first.metrics.get(metric),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'std': np.std(values),
                'change': (latest.metrics.get(metric) - first.metrics.get(metric)) / first.metrics.get(metric)
                          if first.metrics.get(metric) != 0 else float('inf')
            }
        
        # 성능 저하 감지
        degradation_alerts = self._check_performance_degradation(model_id)
        
        # 보고서 생성
        report = {
            'model_id': model_id,
            'report_generated': datetime.now(),
            'period': {
                'start': first.timestamp,
                'end': latest.timestamp,
                'days': (latest.timestamp - first.timestamp).days
            },
            'snapshots_count': len(filtered_history),
            'latest_metrics': latest.metrics,
            'metrics_stats': metrics_stats,
            'degradation_alerts': degradation_alerts
        }
        
        # 트렌드 그래프 생성
        trend_graphs = {}
        for metric in metrics_stats:
            save_path = os.path.join(
                self.save_dir, 'visualizations', 
                f"{model_id}_{metric}_trend_report.{self.config.get('plot_format', 'png')}"
            )
            
            graph_path = self.plot_performance_trend(
                model_id, metric_name=metric,
                start_date=start_date, end_date=end_date,
                show_plot=False, save_path=save_path
            )
            
            if graph_path:
                trend_graphs[metric] = graph_path
        
        report['trend_graphs'] = trend_graphs
        
        return report 