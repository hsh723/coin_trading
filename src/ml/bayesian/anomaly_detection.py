import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Tuple, Optional, Any, Callable
import logging
from scipy import stats
from datetime import datetime, timedelta
import os

# 로깅 설정
logger = logging.getLogger(__name__)

class BayesianAnomalyDetector:
    """
    베이지안 시계열 모델 기반 이상치 탐지 클래스
    
    주요 기능:
    - 모델 예측과 실제 관측치 간의 차이 분석
    - 이상치 점수 계산
    - 이상치 임계값 설정
    - 이상치 시각화
    - 이상치 설명 및 분석
    """
    
    def __init__(self, 
                 model: Any,
                 threshold: float = 0.95,
                 window_size: int = 10,
                 min_anomaly_score: float = 0.5,
                 save_dir: str = "./anomaly_detection"):
        """
        이상치 탐지기 초기화
        
        Args:
            model: 베이지안 시계열 모델
            threshold: 이상치 임계값 (0-1 사이)
            window_size: 이동 윈도우 크기
            min_anomaly_score: 최소 이상치 점수
            save_dir: 결과 저장 디렉토리
        """
        self.model = model
        self.threshold = threshold
        self.window_size = window_size
        self.min_anomaly_score = min_anomaly_score
        self.save_dir = save_dir
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"베이지안 이상치 탐지기 초기화: 임계값={threshold}, 윈도우 크기={window_size}")
    
    def _calculate_residuals(self, 
                           observed: Union[np.ndarray, pd.Series], 
                           predicted: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        관측치와 예측치 간의 잔차 계산
        
        Args:
            observed: 실제 관측치
            predicted: 모델 예측치
            
        Returns:
            잔차 배열
        """
        if isinstance(observed, pd.Series):
            observed = observed.values
        if isinstance(predicted, pd.Series):
            predicted = predicted.values
        
        return observed - predicted
    
    def _calculate_anomaly_scores(self, 
                                residuals: np.ndarray,
                                window_size: Optional[int] = None) -> np.ndarray:
        """
        잔차를 기반으로 이상치 점수 계산
        
        Args:
            residuals: 잔차 배열
            window_size: 이동 윈도우 크기 (None인 경우 기본값 사용)
            
        Returns:
            이상치 점수 배열
        """
        if window_size is None:
            window_size = self.window_size
        
        n = len(residuals)
        scores = np.zeros(n)
        
        # 이동 윈도우 기반 이상치 점수 계산
        for i in range(n):
            # 윈도우 시작 및 끝 인덱스 계산
            start = max(0, i - window_size + 1)
            window_residuals = residuals[start:i+1]
            
            # 윈도우 내 잔차의 통계량 계산
            mean = np.mean(window_residuals)
            std = np.std(window_residuals)
            
            if std > 0:
                # Z-점수 계산
                z_score = abs((residuals[i] - mean) / std)
                # Z-점수를 0-1 사이의 점수로 변환
                score = 1 - np.exp(-z_score)
                scores[i] = score
            else:
                scores[i] = 0
        
        return scores
    
    def detect_anomalies(self, 
                        observed: Union[np.ndarray, pd.Series],
                        predicted: Union[np.ndarray, pd.Series],
                        return_scores: bool = False) -> Dict[str, Any]:
        """
        관측치와 예측치를 비교하여 이상치 탐지
        
        Args:
            observed: 실제 관측치
            predicted: 모델 예측치
            return_scores: 이상치 점수 반환 여부
            
        Returns:
            이상치 탐지 결과 딕셔너리
        """
        # 잔차 계산
        residuals = self._calculate_residuals(observed, predicted)
        
        # 이상치 점수 계산
        scores = self._calculate_anomaly_scores(residuals)
        
        # 이상치 임계값 적용
        anomalies = scores > self.threshold
        
        # 결과 딕셔너리 생성
        result = {
            'is_anomaly': anomalies,
            'residuals': residuals,
            'scores': scores if return_scores else None,
            'threshold': self.threshold
        }
        
        return result
    
    def analyze_anomalies(self, 
                         observed: Union[np.ndarray, pd.Series],
                         predicted: Union[np.ndarray, pd.Series],
                         timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None) -> Dict[str, Any]:
        """
        이상치 상세 분석
        
        Args:
            observed: 실제 관측치
            predicted: 모델 예측치
            timestamps: 시간 인덱스 (옵션)
            
        Returns:
            이상치 분석 결과 딕셔너리
        """
        # 이상치 탐지
        detection_result = self.detect_anomalies(observed, predicted, return_scores=True)
        
        # 이상치 인덱스 추출
        anomaly_indices = np.where(detection_result['is_anomaly'])[0]
        
        # 이상치 분석 결과 딕셔너리
        analysis = {
            'n_anomalies': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'anomaly_scores': detection_result['scores'][anomaly_indices],
            'anomaly_residuals': detection_result['residuals'][anomaly_indices],
            'anomaly_details': []
        }
        
        # 각 이상치에 대한 상세 정보 수집
        for idx in anomaly_indices:
            anomaly_info = {
                'index': idx,
                'observed_value': observed[idx],
                'predicted_value': predicted[idx],
                'residual': detection_result['residuals'][idx],
                'score': detection_result['scores'][idx]
            }
            
            # 시간 정보가 있는 경우 추가
            if timestamps is not None:
                anomaly_info['timestamp'] = timestamps[idx]
            
            analysis['anomaly_details'].append(anomaly_info)
        
        return analysis
    
    def plot_anomalies(self, 
                      observed: Union[np.ndarray, pd.Series],
                      predicted: Union[np.ndarray, pd.Series],
                      timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        이상치 시각화
        
        Args:
            observed: 실제 관측치
            predicted: 모델 예측치
            timestamps: 시간 인덱스 (옵션)
            save_path: 저장 경로 (옵션)
            
        Returns:
            matplotlib Figure 객체
        """
        # 이상치 탐지
        detection_result = self.detect_anomalies(observed, predicted, return_scores=True)
        
        # 시각화 설정
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 시간 인덱스 설정
        if timestamps is None:
            x = np.arange(len(observed))
        else:
            x = timestamps
        
        # 관측치와 예측치 플롯
        ax1.plot(x, observed, label='관측치', color='blue', alpha=0.7)
        ax1.plot(x, predicted, label='예측치', color='red', alpha=0.7)
        
        # 이상치 표시
        anomaly_indices = np.where(detection_result['is_anomaly'])[0]
        ax1.scatter(x[anomaly_indices], observed[anomaly_indices], 
                   color='red', s=100, label='이상치', zorder=5)
        
        ax1.set_title('관측치 vs 예측치 및 이상치')
        ax1.set_ylabel('값')
        ax1.legend()
        ax1.grid(True)
        
        # 이상치 점수 플롯
        ax2.plot(x, detection_result['scores'], label='이상치 점수', color='green')
        ax2.axhline(y=self.threshold, color='red', linestyle='--', label='임계값')
        
        ax2.set_title('이상치 점수')
        ax2.set_xlabel('시간' if timestamps is not None else '인덱스')
        ax2.set_ylabel('이상치 점수')
        ax2.legend()
        ax2.grid(True)
        
        # x축 레이블 회전 (시간 인덱스인 경우)
        if timestamps is not None:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 저장 경로가 지정된 경우 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"이상치 시각화 저장: {save_path}")
        
        return fig
    
    def generate_report(self, 
                       observed: Union[np.ndarray, pd.Series],
                       predicted: Union[np.ndarray, pd.Series],
                       timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        이상치 분석 보고서 생성
        
        Args:
            observed: 실제 관측치
            predicted: 모델 예측치
            timestamps: 시간 인덱스 (옵션)
            save_path: 저장 경로 (옵션)
            
        Returns:
            보고서 딕셔너리
        """
        # 이상치 분석
        analysis = self.analyze_anomalies(observed, predicted, timestamps)
        
        # 보고서 딕셔너리 생성
        report = {
            'summary': {
                'n_anomalies': analysis['n_anomalies'],
                'anomaly_rate': analysis['n_anomalies'] / len(observed),
                'threshold': self.threshold,
                'window_size': self.window_size,
                'min_anomaly_score': self.min_anomaly_score
            },
            'anomaly_statistics': {
                'mean_score': np.mean(analysis['anomaly_scores']),
                'max_score': np.max(analysis['anomaly_scores']),
                'mean_residual': np.mean(analysis['anomaly_residuals']),
                'max_residual': np.max(np.abs(analysis['anomaly_residuals']))
            },
            'anomaly_details': analysis['anomaly_details']
        }
        
        # 저장 경로가 지정된 경우 JSON 파일로 저장
        if save_path:
            import json
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"이상치 분석 보고서 저장: {save_path}")
        
        return report
    
    def update_threshold(self, 
                        observed: Union[np.ndarray, pd.Series],
                        predicted: Union[np.ndarray, pd.Series],
                        target_anomaly_rate: float = 0.05) -> float:
        """
        목표 이상치 비율에 맞게 임계값 조정
        
        Args:
            observed: 실제 관측치
            predicted: 모델 예측치
            target_anomaly_rate: 목표 이상치 비율
            
        Returns:
            조정된 임계값
        """
        # 잔차 계산
        residuals = self._calculate_residuals(observed, predicted)
        
        # 이상치 점수 계산
        scores = self._calculate_anomaly_scores(residuals)
        
        # 목표 이상치 비율에 맞는 임계값 계산
        threshold = np.percentile(scores, (1 - target_anomaly_rate) * 100)
        
        # 임계값 업데이트
        self.threshold = threshold
        
        logger.info(f"임계값 업데이트: {threshold:.4f} (목표 이상치 비율: {target_anomaly_rate})")
        return threshold 