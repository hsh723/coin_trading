import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression, RFE
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class AdaptiveFeatureSelector:
    """적응형 특성 선택 메커니즘
    
    시간에 따라 변화하는 데이터에서 중요한 특성을 동적으로 선택하는 클래스:
    - 다양한 특성 중요도 평가 방법 지원
    - 교차 검증을 통한 최적 특성 부분집합 선택
    - 시간에 따른 특성 중요도 변화 추적
    - 상황에 따른 자동 특성 선택
    """
    
    SELECTION_METHODS = [
        'random_forest', 'gradient_boosting', 'lasso', 
        'mutual_info', 'f_regression', 'recursive_elimination'
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 설정 파라미터
        """
        # 기본 설정
        self.config = {
            'base_features': [],               # 기본 포함되는 특성 (항상 선택)
            'selection_method': 'random_forest',  # 특성 선택 방법
            'importance_threshold': 0.01,      # 중요도 임계값 (1%)
            'min_features': 5,                 # 최소 특성 수
            'max_features': 20,                # 최대 특성 수
            'cv_folds': 5,                     # 교차 검증 폴드 수
            'use_validation': True,            # 검증 데이터 사용 여부
            'track_importance_history': True,  # 중요도 변화 추적 여부
            'auto_adjust_threshold': True,     # 임계값 자동 조정 여부
            'auto_adjust_interval': 10,        # 자동 조정 간격 (선택 횟수)
            'save_dir': 'models/feature_selection'  # 저장 디렉토리
        }
        
        # 사용자 설정 업데이트
        if config:
            self.config.update(config)
        
        # 디렉토리 생성
        if self.config['track_importance_history'] or self.config['auto_adjust_threshold']:
            os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # 특성 중요도 기록
        self.feature_importances = {}
        self.importance_history = []
        self.selected_features_history = []
        self.selection_count = 0
        self.last_validation_score = None
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                       n_features: Optional[int] = None,
                       method: Optional[str] = None) -> List[str]:
        """최적 특성 선택
        
        Args:
            X: 특성 데이터프레임
            y: 타겟 변수
            validation_data: 검증 데이터 (X_val, y_val)
            n_features: 선택할 특성 수 (None이면 자동 결정)
            method: 특성 선택 방법 (None이면 config에서 가져옴)
            
        Returns:
            선택된 특성 목록
        """
        if method is None:
            method = self.config['selection_method']
        
        if method not in self.SELECTION_METHODS:
            raise ValueError(f"지원하지 않는 특성 선택 방법: {method}. "
                           f"지원되는 방법: {self.SELECTION_METHODS}")
        
        # 특성 중요도 계산
        importances = self._calculate_feature_importance(X, y, method)
        
        # 중요도 기준 특성 정렬
        sorted_features = [
            (feature, importance) 
            for feature, importance in sorted(
                importances.items(), key=lambda x: x[1], reverse=True
            )
        ]
        
        # 중요도 기록
        self.feature_importances = dict(sorted_features)
        self._update_importance_history()
        
        # 중요도 임계값
        threshold = self.config['importance_threshold']
        
        # 임계값 이상 특성 선택
        selected_features = [
            feature for feature, importance in sorted_features
            if importance >= threshold
        ]
        
        # 기본 특성 추가
        for feature in self.config['base_features']:
            if feature in X.columns and feature not in selected_features:
                selected_features.append(feature)
        
        # 최소/최대 특성 수 제약 처리
        if n_features is not None:
            # 지정된 특성 수로 상위 특성 선택
            selected_features = [f[0] for f in sorted_features[:n_features]]
            # 기본 특성 추가 보장
            for feature in self.config['base_features']:
                if feature in X.columns and feature not in selected_features:
                    # 마지막 특성을 제거하고 기본 특성 추가
                    selected_features = selected_features[:-1] + [feature]
        else:
            # 최소 특성 수 보장
            if len(selected_features) < self.config['min_features']:
                remaining = [f[0] for f in sorted_features if f[0] not in selected_features]
                needed = self.config['min_features'] - len(selected_features)
                selected_features.extend(remaining[:needed])
            
            # 최대 특성 수 제한
            if len(selected_features) > self.config['max_features']:
                # 중요도 순으로 정렬된 특성 중 상위 n개 선택 (기본 특성 보장)
                important_features = [f for f in selected_features if f not in self.config['base_features']]
                important_features = important_features[:self.config['max_features'] - len(self.config['base_features'])]
                selected_features = important_features + [f for f in self.config['base_features'] if f in X.columns]
        
        # 특성 부분집합 최적화 (검증 데이터 사용)
        if self.config['use_validation'] and validation_data is not None:
            selected_features = self._optimize_feature_subset(
                X, y, selected_features, validation_data
            )
        
        # 선택 횟수 증가
        self.selection_count += 1
        
        # 선택된 특성 기록
        self.selected_features_history.append({
            'timestamp': datetime.now(),
            'features': selected_features,
            'method': method,
            'threshold': threshold,
            'validation_score': self.last_validation_score
        })
        
        # 임계값 자동 조정
        if (self.config['auto_adjust_threshold'] and 
            self.selection_count % self.config['auto_adjust_interval'] == 0):
            self._auto_adjust_threshold()
        
        return selected_features
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                                    method: str) -> Dict[str, float]:
        """특성 중요도 계산
        
        Args:
            X: 특성 데이터프레임
            y: 타겟 변수
            method: 특성 선택 방법
            
        Returns:
            특성별 중요도 딕셔너리
        """
        # 특성 이름 목록
        feature_names = X.columns.tolist()
        
        if method == 'random_forest':
            # 랜덤 포레스트 기반 중요도
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            importances = dict(zip(feature_names, model.feature_importances_))
            
        elif method == 'gradient_boosting':
            # 그래디언트 부스팅 기반 중요도
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            importances = dict(zip(feature_names, model.feature_importances_))
            
        elif method == 'lasso':
            # Lasso 회귀 기반 중요도
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = Lasso(alpha=0.01, random_state=42)
            model.fit(X_scaled, y)
            
            # 절대값 가중치를 중요도로 사용
            importances = dict(zip(feature_names, np.abs(model.coef_)))
            
        elif method == 'mutual_info':
            # 상호 정보량 기반 중요도
            # 결측치 처리 (상호 정보량은 결측치를 허용하지 않음)
            X_filled = X.fillna(X.mean())
            mi_scores = mutual_info_regression(X_filled, y)
            importances = dict(zip(feature_names, mi_scores))
            
        elif method == 'f_regression':
            # F-통계량 기반 중요도
            # 결측치 처리
            X_filled = X.fillna(X.mean())
            f_scores, _ = f_regression(X_filled, y)
            importances = dict(zip(feature_names, f_scores))
            
        elif method == 'recursive_elimination':
            # 재귀적 특성 제거 기반 중요도
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            rfe = RFE(estimator=model, n_features_to_select=len(feature_names)//2)
            rfe.fit(X, y)
            
            # 특성 순위의 역수를 중요도로 사용
            ranks = rfe.ranking_
            max_rank = np.max(ranks) + 1
            importances = dict(zip(feature_names, [max_rank - r for r in ranks]))
        
        # 중요도 정규화 (합이 1이 되도록)
        total_importance = sum(importances.values())
        if total_importance > 0:
            importances = {f: i/total_importance for f, i in importances.items()}
        
        return importances
    
    def _optimize_feature_subset(self, X: pd.DataFrame, y: pd.Series, 
                               features: List[str],
                               validation_data: Tuple[pd.DataFrame, pd.Series]) -> List[str]:
        """교차 검증을 통한 특성 부분집합 최적화
        
        Args:
            X: 특성 데이터프레임
            y: 타겟 변수
            features: 선택된 특성 목록
            validation_data: 검증 데이터 (X_val, y_val)
            
        Returns:
            최적화된 특성 목록
        """
        X_val, y_val = validation_data
        best_score = -np.inf
        best_features = features
        
        # 기본 특성은 항상 포함
        base_features = [f for f in self.config['base_features'] if f in X.columns]
        
        # 후보 특성 (기본 특성 제외)
        candidate_features = [f for f in features if f not in base_features]
        
        if len(candidate_features) <= 1:
            # 후보 특성이 1개 이하면 최적화 불필요
            self.last_validation_score = self._evaluate_features(
                X, y, X_val, y_val, features
            )
            return features
        
        # 전방 선택법 (Forward Selection) 사용
        selected = base_features.copy()
        remaining = candidate_features.copy()
        
        while remaining:
            best_new_feature = None
            best_new_score = best_score
            
            # 남은 특성 중 최선의 특성 선택
            for feature in remaining:
                current_features = selected + [feature]
                score = self._evaluate_features(
                    X, y, X_val, y_val, current_features
                )
                
                if score > best_new_score:
                    best_new_score = score
                    best_new_feature = feature
            
            # 성능 향상이 있으면 특성 추가
            if best_new_feature and best_new_score > best_score:
                best_score = best_new_score
                selected.append(best_new_feature)
                remaining.remove(best_new_feature)
            else:
                # 성능 향상이 없으면 중단
                break
        
        self.last_validation_score = best_score
        return selected
    
    def _evaluate_features(self, X: pd.DataFrame, y: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         features: List[str]) -> float:
        """특성 부분집합 평가
        
        Args:
            X: 훈련 특성 데이터프레임
            y: 훈련 타겟 변수
            X_val: 검증 특성 데이터프레임
            y_val: 검증 타겟 변수
            features: 평가할 특성 목록
            
        Returns:
            검증 점수
        """
        # 특성 부분집합 선택
        X_subset = X[features]
        X_val_subset = X_val[features]
        
        # 평가 모델 (랜덤 포레스트 사용)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # 훈련
        model.fit(X_subset, y)
        
        # 검증 점수 계산
        score = model.score(X_val_subset, y_val)
        return score
    
    def _update_importance_history(self) -> None:
        """특성 중요도 히스토리 업데이트"""
        if not self.config['track_importance_history']:
            return
        
        self.importance_history.append({
            'timestamp': datetime.now(),
            'importances': self.feature_importances
        })
    
    def _auto_adjust_threshold(self) -> None:
        """중요도 임계값 자동 조정"""
        if len(self.selected_features_history) < 2:
            return
        
        # 최근 n개 선택 결과 가져오기
        n = self.config['auto_adjust_interval']
        recent_history = self.selected_features_history[-n:]
        
        # 검증 점수가 있는 경우, 검증 점수 추이 분석
        has_scores = all(h.get('validation_score') is not None for h in recent_history)
        
        if has_scores:
            scores = [h['validation_score'] for h in recent_history]
            # 점수 변화 추세 확인
            if len(scores) >= 3:
                # 최근 점수 감소 추세인지 확인
                recent_trend = scores[-1] < scores[-2] < scores[-3]
                
                if recent_trend:
                    # 중요도 임계값 낮추기
                    self.config['importance_threshold'] *= 0.9
                    self.logger.info(f"검증 점수 감소 추세로 임계값 낮춤: {self.config['importance_threshold']:.6f}")
                elif scores[-1] > scores[-2]:
                    # 최근 점수 증가면 임계값 유지
                    pass
                else:
                    # 임계값 약간 높이기
                    self.config['importance_threshold'] *= 1.05
                    self.logger.info(f"검증 점수 개선을 위해 임계값 높임: {self.config['importance_threshold']:.6f}")
        else:
            # 검증 점수가 없는 경우 특성 수 기반 조정
            feature_counts = [len(h['features']) for h in recent_history]
            avg_count = sum(feature_counts) / len(feature_counts)
            
            # 특성 수가 너무 많으면 임계값 증가
            if avg_count > self.config['max_features'] * 0.8:
                self.config['importance_threshold'] *= 1.1
                self.logger.info(f"특성 수가 많아 임계값 증가: {self.config['importance_threshold']:.6f}")
            # 특성 수가 너무 적으면 임계값 감소
            elif avg_count < self.config['min_features'] * 1.2:
                self.config['importance_threshold'] *= 0.9
                self.logger.info(f"특성 수가 적어 임계값 감소: {self.config['importance_threshold']:.6f}")
        
        # 임계값 범위 제한
        self.config['importance_threshold'] = max(0.001, min(0.2, self.config['importance_threshold']))
    
    def get_feature_importances(self) -> Dict[str, float]:
        """현재 특성 중요도 반환"""
        return self.feature_importances
    
    def get_importance_history(self) -> List[Dict[str, Any]]:
        """특성 중요도 변화 기록 반환"""
        return self.importance_history
    
    def get_selected_features_history(self) -> List[Dict[str, Any]]:
        """선택된 특성 변화 기록 반환"""
        return self.selected_features_history
    
    def plot_feature_importances(self, top_n: int = 15, 
                               show_plot: bool = True,
                               save_path: Optional[str] = None) -> None:
        """특성 중요도 시각화
        
        Args:
            top_n: 표시할 상위 특성 수
            show_plot: 그래프 표시 여부
            save_path: 그래프 저장 경로
        """
        if not self.feature_importances:
            raise ValueError("특성 중요도가 계산되지 않았습니다")
        
        # 중요도 기준 특성 정렬
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 상위 n개 특성 선택
        top_features = sorted_features[:top_n]
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        plt.figure(figsize=(10, 8))
        
        # 수평 막대 그래프
        sns.barplot(x=importances, y=features, palette='viridis')
        
        plt.title('특성 중요도')
        plt.xlabel('중요도')
        plt.ylabel('특성')
        plt.tight_layout()
        
        # 그래프 저장
        if save_path:
            plt.savefig(save_path)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_importance_trend(self, features: Optional[List[str]] = None,
                            top_n: int = 5,
                            show_plot: bool = True,
                            save_path: Optional[str] = None) -> None:
        """특성 중요도 변화 추이 시각화
        
        Args:
            features: 표시할 특성 목록 (None이면 상위 n개 특성)
            top_n: 표시할 상위 특성 수
            show_plot: 그래프 표시 여부
            save_path: 그래프 저장 경로
        """
        if not self.importance_history:
            raise ValueError("특성 중요도 변화 기록이 없습니다")
        
        # 마지막 중요도 기준 상위 특성 선택
        if features is None:
            last_importances = self.importance_history[-1]['importances']
            sorted_features = sorted(
                last_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            features = [f[0] for f in sorted_features[:top_n]]
        
        # 타임스탬프 및 중요도 추출
        timestamps = [h['timestamp'] for h in self.importance_history]
        
        plt.figure(figsize=(12, 6))
        
        # 각 특성별 중요도 변화 그래프
        for feature in features:
            importances = []
            for h in self.importance_history:
                # 특성이 있는지 확인
                if feature in h['importances']:
                    importances.append(h['importances'][feature])
                else:
                    importances.append(0)
            
            plt.plot(timestamps, importances, 'o-', label=feature)
        
        plt.title('특성 중요도 변화 추이')
        plt.xlabel('시간')
        plt.ylabel('중요도')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 그래프 저장
        if save_path:
            plt.savefig(save_path)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def save(self, filepath: str) -> None:
        """모델 저장
        
        Args:
            filepath: 저장 경로
        """
        data = {
            'config': self.config,
            'feature_importances': self.feature_importances,
            'importance_history': self.importance_history,
            'selected_features_history': self.selected_features_history,
            'selection_count': self.selection_count,
            'last_validation_score': self.last_validation_score
        }
        
        joblib.dump(data, filepath)
        self.logger.info(f"적응형 특성 선택기 저장 완료: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AdaptiveFeatureSelector':
        """저장된 모델 로드
        
        Args:
            filepath: 저장 경로
            
        Returns:
            로드된 모델
        """
        data = joblib.load(filepath)
        
        instance = cls(config=data['config'])
        instance.feature_importances = data['feature_importances']
        instance.importance_history = data['importance_history']
        instance.selected_features_history = data['selected_features_history']
        instance.selection_count = data['selection_count']
        instance.last_validation_score = data['last_validation_score']
        
        return instance 