#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
시장 상태 분류 모델
시장 상태를 상승(Bullish), 하락(Bearish), 횡보(Sideways) 등으로 분류
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime
import json

class MarketStateClassifier:
    """시장 상태 분류 모델"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        초기화
        
        Args:
            config: 설정 파라미터
        """
        self.config = config or {
            'class_thresholds': {
                'bullish': 0.02,   # 2% 이상 상승
                'bearish': -0.02,  # 2% 이상 하락
                'sideways': 0.005  # -0.5% ~ 0.5% 횡보
            },
            'prediction_period': 24,  # 24시간(1일) 후 예측
            'model_type': 'random_forest',  # 기본 모델 유형
            'feature_window': 14,  # 특성 생성을 위한 기간
            'model_save_path': 'models/market_state/'
        }
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, data: pd.DataFrame, 
                    features: Optional[List[str]] = None,
                    labeled: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        데이터 준비 및 특성 생성
        
        Args:
            data: 원시 데이터
            features: 사용할 특성 목록
            labeled: 레이블 생성 여부
            
        Returns:
            특성 데이터프레임, 레이블 시리즈(있는 경우)
        """
        # 데이터 복사
        df = data.copy()
        
        # 결측치 제거
        df = df.dropna()
        
        # 기본 특성 목록
        if features is None:
            features = ['open', 'high', 'low', 'close', 'volume']
        
        # 추가 특성 생성
        feature_window = self.config.get('feature_window', 14)
        
        # 가격 변동성
        df['volatility'] = df['high'].div(df['low']) - 1
        
        # 이동평균선 대비 가격
        df['ma14_ratio'] = df['close'] / df['close'].rolling(window=14).mean()
        df['ma30_ratio'] = df['close'] / df['close'].rolling(window=30).mean()
        
        # 거래량 특성
        df['volume_ma14_ratio'] = df['volume'] / df['volume'].rolling(window=14).mean()
        
        # 수익률 특성
        df['return_1d'] = df['close'].pct_change(periods=1)
        df['return_3d'] = df['close'].pct_change(periods=3)
        df['return_7d'] = df['close'].pct_change(periods=7)
        
        # 방향성 특성
        df['direction_1d'] = np.where(df['return_1d'] > 0, 1, -1)
        df['direction_3d'] = np.where(df['return_3d'] > 0, 1, -1)
        df['direction_7d'] = np.where(df['return_7d'] > 0, 1, -1)
        
        # RSI 지표
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD 지표
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 레이블 생성 (선택적)
        labels = None
        if labeled:
            prediction_period = self.config.get('prediction_period', 24)
            thresholds = self.config.get('class_thresholds', {
                'bullish': 0.02,
                'bearish': -0.02,
                'sideways': 0.005
            })
            
            # 미래 수익률 계산
            future_returns = df['close'].pct_change(periods=prediction_period).shift(-prediction_period)
            
            # 레이블 생성
            labels = pd.Series(index=df.index, dtype='object')
            labels[(future_returns >= thresholds['bullish'])] = 'bullish'
            labels[(future_returns <= thresholds['bearish'])] = 'bearish'
            labels[(future_returns.abs() <= thresholds['sideways'])] = 'sideways'
            labels[labels.isna()] = 'neutral'  # 기타는 중립으로 분류
            
            # 분류 개수 확인
            print("클래스 분포:")
            print(labels.value_counts())
        
        # 사용할 특성 선택
        all_features = [
            'close', 'high', 'low', 'volume', 'volatility', 'ma14_ratio', 'ma30_ratio',
            'volume_ma14_ratio', 'return_1d', 'return_3d', 'return_7d',
            'direction_1d', 'direction_3d', 'direction_7d', 'rsi', 'macd', 'macd_hist'
        ]
        feature_df = df[all_features].copy()
        
        # 결측치 제거
        feature_df = feature_df.dropna()
        
        if labeled:
            # 레이블도 결측치 제거에 맞춰 조정
            labels = labels.loc[feature_df.index]
        
        return feature_df, labels
    
    def train(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
        """
        모델 훈련
        
        Args:
            features: 특성 데이터프레임
            labels: 레이블 시리즈
            
        Returns:
            성능 지표
        """
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 모델 선택
        model_type = self.config.get('model_type', 'random_forest')
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
        elif model_type == 'xgboost':
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'svm':
            model = SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {model_type}")
        
        # 모델 훈련
        model.fit(X_train_scaled, y_train)
        self.model = model
        
        # 성능 평가
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # 특성 중요도 (가능한 경우)
        feature_importances = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            features_list = features.columns.tolist()
            
            for i in range(len(features_list)):
                feature_name = features_list[indices[i]]
                feature_importances[feature_name] = float(importances[indices[i]])
        
        # 결과 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # 특성 중요도 시각화 (가능한 경우)
        if feature_importances:
            plt.figure(figsize=(12, 6))
            sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            features_sorted = [x[0] for x in sorted_importances]
            importances_sorted = [x[1] for x in sorted_importances]
            
            plt.barh(range(len(features_sorted)), importances_sorted, align='center')
            plt.yticks(range(len(features_sorted)), features_sorted)
            plt.title('Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
        
        # 결과 반환
        metrics = {
            'accuracy': accuracy,
            'report': report,
            'feature_importances': feature_importances
        }
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        시장 상태 예측
        
        Args:
            features: 특성 데이터프레임
            
        Returns:
            예측 결과 데이터프레임
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 특성 스케일링
        scaled_features = self.scaler.transform(features)
        
        # 예측
        predictions = self.model.predict(scaled_features)
        
        # 확률 예측 (가능한 경우)
        probabilities = pd.DataFrame(index=features.index)
        if hasattr(self.model, 'predict_proba'):
            proba_matrix = self.model.predict_proba(scaled_features)
            for i, class_name in enumerate(self.model.classes_):
                probabilities[f'prob_{class_name}'] = proba_matrix[:, i]
        
        # 결과 데이터프레임 생성
        results = pd.DataFrame({
            'predicted_state': predictions
        }, index=features.index)
        
        # 확률 추가
        if not probabilities.empty:
            results = pd.concat([results, probabilities], axis=1)
        
        return results
    
    def save(self, path: Optional[str] = None) -> str:
        """
        모델 저장
        
        Args:
            path: 저장 경로
            
        Returns:
            모델 저장 경로
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        # 저장 경로 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.config.get('model_type', 'random_forest')
        model_dir = path or self.config.get('model_save_path', 'models/market_state/')
        
        # 디렉토리 생성
        os.makedirs(model_dir, exist_ok=True)
        
        # 모델 저장
        model_path = os.path.join(model_dir, f"{model_type}_{timestamp}.pkl")
        joblib.dump(self.model, model_path)
        
        # 스케일러 저장
        scaler_path = os.path.join(model_dir, f"{model_type}_{timestamp}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # 설정 저장
        config_path = os.path.join(model_dir, f"{model_type}_{timestamp}_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
            
        print(f"모델이 저장되었습니다: {model_path}")
        
        return model_path
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str, config_path: Optional[str] = None) -> 'MarketStateClassifier':
        """
        저장된 모델 로드
        
        Args:
            model_path: 모델 파일 경로
            scaler_path: 스케일러 파일 경로
            config_path: 설정 파일 경로
            
        Returns:
            로드된 모델 인스턴스
        """
        # 설정 로드
        config = None
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # 인스턴스 생성
        classifier = cls(config)
        
        # 모델 로드
        classifier.model = joblib.load(model_path)
        
        # 스케일러 로드
        classifier.scaler = joblib.load(scaler_path)
        
        return classifier 