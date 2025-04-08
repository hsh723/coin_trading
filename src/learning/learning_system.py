import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from database import Database

class LearningSystem:
    def __init__(self, db: Database):
        self.logger = logging.getLogger(__name__)
        self.db = db
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_training_data(self, trades: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """학습 데이터 준비"""
        X = []
        y = []
        
        for trade in trades:
            # 기술적 지표
            indicators = trade['entry_indicators']
            
            # 특징 벡터 생성
            features = [
                indicators['trend']['ma20'],
                indicators['trend']['ma50'],
                indicators['trend']['ma200'],
                indicators['trend']['macd']['macd'],
                indicators['trend']['macd']['signal'],
                indicators['trend']['adx'],
                indicators['momentum']['rsi'],
                indicators['momentum']['stochastic']['k'],
                indicators['momentum']['stochastic']['d'],
                indicators['momentum']['cci'],
                indicators['volatility']['bollinger_bands']['upper'],
                indicators['volatility']['bollinger_bands']['lower'],
                indicators['volatility']['atr'],
                indicators['volume']['obv'],
                indicators['volume']['mfi']
            ]
            
            # 레이블 생성 (수익성 있는 거래인지)
            profit = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
            label = 1 if profit > 0 else 0
            
            X.append(features)
            y.append(label)
            
        return np.array(X), np.array(y)
        
    def train_model(self, trades: List[Dict]) -> None:
        """모델 학습"""
        X, y = self.prepare_training_data(trades)
        
        # 데이터 스케일링
        X_scaled = self.scaler.fit_transform(X)
        
        # 모델 학습
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        self.logger.info("모델 학습 완료")
        
    def predict_trade(self, indicators: Dict) -> Dict:
        """거래 예측"""
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        # 특징 벡터 생성
        features = [
            indicators['trend']['ma20'],
            indicators['trend']['ma50'],
            indicators['trend']['ma200'],
            indicators['trend']['macd']['macd'],
            indicators['trend']['macd']['signal'],
            indicators['trend']['adx'],
            indicators['momentum']['rsi'],
            indicators['momentum']['stochastic']['k'],
            indicators['momentum']['stochastic']['d'],
            indicators['momentum']['cci'],
            indicators['volatility']['bollinger_bands']['upper'],
            indicators['volatility']['bollinger_bands']['lower'],
            indicators['volatility']['atr'],
            indicators['volume']['obv'],
            indicators['volume']['mfi']
        ]
        
        # 데이터 스케일링
        features_scaled = self.scaler.transform([features])
        
        # 예측
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': prediction,
            'probability': probability[1],
            'confidence': max(probability)
        }
        
    def analyze_trade(self, trade: Dict) -> Dict:
        """거래 분석"""
        analysis = {
            'entry_analysis': self._analyze_entry(trade),
            'exit_analysis': self._analyze_exit(trade),
            'improvements': self._suggest_improvements(trade)
        }
        
        return analysis
        
    def _analyze_entry(self, trade: Dict) -> Dict:
        """진입 분석"""
        entry_analysis = {
            'timing': None,
            'price_level': None,
            'indicator_alignment': None,
            'strengths': [],
            'weaknesses': []
        }
        
        indicators = trade['entry_indicators']
        
        # 타이밍 분석
        if (indicators['trend']['ma20'] > indicators['trend']['ma50'] and
            indicators['momentum']['rsi'] < 70):
            entry_analysis['timing'] = 'good'
            entry_analysis['strengths'].append('상승 추세에서 과매수되지 않은 상태')
        else:
            entry_analysis['timing'] = 'poor'
            entry_analysis['weaknesses'].append('추세와 모멘텀 정렬이 좋지 않음')
            
        # 가격 레벨 분석
        if trade['entry_price'] < indicators['volatility']['bollinger_bands']['lower']:
            entry_analysis['price_level'] = 'good'
            entry_analysis['strengths'].append('볼린저 밴드 하단에서 매수')
        else:
            entry_analysis['price_level'] = 'poor'
            entry_analysis['weaknesses'].append('매수 가격이 최적이 아님')
            
        # 지표 정렬 분석
        aligned_indicators = 0
        total_indicators = 0
        
        if indicators['trend']['macd']['macd'] > indicators['trend']['macd']['signal']:
            aligned_indicators += 1
        total_indicators += 1
        
        if indicators['momentum']['rsi'] < 70:
            aligned_indicators += 1
        total_indicators += 1
        
        if indicators['volume']['obv'] > indicators['volume']['obv'].shift(1):
            aligned_indicators += 1
        total_indicators += 1
        
        alignment_ratio = aligned_indicators / total_indicators
        entry_analysis['indicator_alignment'] = 'good' if alignment_ratio > 0.7 else 'poor'
        
        return entry_analysis
        
    def _analyze_exit(self, trade: Dict) -> Dict:
        """청산 분석"""
        exit_analysis = {
            'timing': None,
            'price_level': None,
            'profit_taking': None,
            'stop_loss': None,
            'strengths': [],
            'weaknesses': []
        }
        
        # 타이밍 분석
        profit = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
        
        if profit > 0:
            exit_analysis['timing'] = 'good'
            exit_analysis['strengths'].append('수익 실현')
        else:
            exit_analysis['timing'] = 'poor'
            exit_analysis['weaknesses'].append('손실 발생')
            
        # 가격 레벨 분석
        if trade['exit_price'] > trade['entry_price'] * 1.02:
            exit_analysis['price_level'] = 'good'
            exit_analysis['strengths'].append('2% 이상 수익 실현')
        else:
            exit_analysis['price_level'] = 'poor'
            exit_analysis['weaknesses'].append('수익 목표 미달')
            
        # 이익 실현 분석
        if trade.get('take_profit_hit', False):
            exit_analysis['profit_taking'] = 'good'
            exit_analysis['strengths'].append('이익 실현 목표 달성')
        else:
            exit_analysis['profit_taking'] = 'poor'
            exit_analysis['weaknesses'].append('이익 실현 목표 미달')
            
        # 손절 분석
        if trade.get('stop_loss_hit', False):
            exit_analysis['stop_loss'] = 'good'
            exit_analysis['strengths'].append('손절 기준 준수')
        else:
            exit_analysis['stop_loss'] = 'poor'
            exit_analysis['weaknesses'].append('손절 기준 미준수')
            
        return exit_analysis
        
    def _suggest_improvements(self, trade: Dict) -> List[str]:
        """개선점 제안"""
        improvements = []
        
        # 진입 개선점
        entry_analysis = self._analyze_entry(trade)
        if entry_analysis['timing'] == 'poor':
            improvements.append('진입 타이밍 개선 필요')
        if entry_analysis['price_level'] == 'poor':
            improvements.append('진입 가격 레벨 최적화 필요')
        if entry_analysis['indicator_alignment'] == 'poor':
            improvements.append('지표 정렬 개선 필요')
            
        # 청산 개선점
        exit_analysis = self._analyze_exit(trade)
        if exit_analysis['timing'] == 'poor':
            improvements.append('청산 타이밍 개선 필요')
        if exit_analysis['price_level'] == 'poor':
            improvements.append('수익 목표 재설정 필요')
        if exit_analysis['profit_taking'] == 'poor':
            improvements.append('이익 실현 전략 개선 필요')
        if exit_analysis['stop_loss'] == 'poor':
            improvements.append('손절 전략 개선 필요')
            
        return improvements
        
    def update_model(self, new_trades: List[Dict]) -> None:
        """모델 업데이트"""
        if not self.is_trained:
            self.train_model(new_trades)
            return
            
        # 기존 모델에 새로운 데이터로 추가 학습
        X, y = self.prepare_training_data(new_trades)
        X_scaled = self.scaler.transform(X)
        
        self.model.fit(X_scaled, y)
        self.logger.info("모델 업데이트 완료")
        
    def get_feature_importance(self) -> Dict:
        """특징 중요도 분석"""
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        feature_names = [
            'MA20', 'MA50', 'MA200',
            'MACD', 'MACD Signal', 'ADX',
            'RSI', 'Stochastic K', 'Stochastic D', 'CCI',
            'BB Upper', 'BB Lower', 'ATR',
            'OBV', 'MFI'
        ]
        
        importance = self.model.feature_importances_
        sorted_idx = importance.argsort()[::-1]
        
        return {
            feature_names[i]: importance[i] for i in sorted_idx
        }
        
    def evaluate_model(self, test_trades: List[Dict]) -> Dict:
        """모델 평가"""
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        X, y = self.prepare_training_data(test_trades)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # 정확도 계산
        accuracy = np.mean(predictions == y)
        
        # 정밀도, 재현율 계산
        true_positives = np.sum((predictions == 1) & (y == 1))
        false_positives = np.sum((predictions == 1) & (y == 0))
        false_negatives = np.sum((predictions == 0) & (y == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        } 