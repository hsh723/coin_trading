"""
자가 학습 모듈
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import optuna
import joblib
import os

class SelfLearningSystem:
    """자가 학습 시스템 클래스"""
    
    def __init__(self, 
                db_manager,
                model_dir: str = "models",
                n_trials: int = 100):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            model_dir (str): 모델 저장 디렉토리
            n_trials (int): 최적화 시도 횟수
        """
        self.db_manager = db_manager
        self.model_dir = model_dir
        self.n_trials = n_trials
        self.logger = logging.getLogger(__name__)
        self._init_model_dir()
        
    def _init_model_dir(self):
        """모델 디렉토리 초기화"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"모델 디렉토리 생성 실패: {str(e)}")
            raise
            
    def _prepare_training_data(self, 
                             lookback_days: int = 30) -> Tuple[pd.DataFrame, pd.Series]:
        """
        학습 데이터 준비
        
        Args:
            lookback_days (int): 학습 데이터 기간
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 특성 데이터와 타겟 데이터
        """
        try:
            # 거래 기록 조회
            trades = self.db_manager.get_trades(
                start_time=datetime.now() - timedelta(days=lookback_days)
            )
            
            if len(trades) < 10:
                self.logger.warning("학습 데이터가 부족합니다.")
                return None, None
                
            # 특성 및 타겟 데이터 생성
            features = []
            targets = []
            
            for trade in trades:
                # 거래 특성
                trade_features = {
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'size': trade['size'],
                    'holding_time': (trade['exit_time'] - trade['entry_time']).total_seconds(),
                    'pnl': trade['pnl'],
                    'pnl_pct': trade['pnl'] / (trade['entry_price'] * trade['size'])
                }
                
                # 시장 특성
                market_data = self.db_manager.get_market_data(
                    symbol=trade['symbol'],
                    start_time=trade['entry_time'],
                    end_time=trade['exit_time']
                )
                
                if market_data is not None:
                    trade_features.update({
                        'volatility': market_data['close'].std(),
                        'volume': market_data['volume'].mean(),
                        'rsi': market_data['rsi'].mean(),
                        'bb_width': (market_data['bb_upper'] - market_data['bb_lower']).mean()
                    })
                
                features.append(trade_features)
                targets.append(trade['pnl_pct'])
            
            return pd.DataFrame(features), pd.Series(targets)
            
        except Exception as e:
            self.logger.error(f"학습 데이터 준비 실패: {str(e)}")
            return None, None
            
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """
        하이퍼파라미터 최적화 목적 함수
        
        Args:
            trial: Optuna trial 객체
            X_train: 학습 데이터 특성
            y_train: 학습 데이터 타겟
            X_val: 검증 데이터 특성
            y_val: 검증 데이터 타겟
            
        Returns:
            float: 검증 데이터 MSE
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }
        
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        return mean_squared_error(y_val, y_pred)
        
    def optimize_strategy(self) -> Dict:
        """
        전략 파라미터 최적화
        
        Returns:
            Dict: 최적화된 파라미터
        """
        try:
            # 학습 데이터 준비
            X, y = self._prepare_training_data()
            if X is None or y is None:
                return None
                
            # 데이터 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 하이퍼파라미터 최적화
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
                n_trials=self.n_trials
            )
            
            # 최적 모델 학습
            best_params = study.best_params
            best_model = RandomForestRegressor(**best_params, random_state=42)
            best_model.fit(X, y)
            
            # 모델 저장
            model_path = os.path.join(
                self.model_dir,
                f"strategy_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
            joblib.dump(best_model, model_path)
            
            # 최적화된 파라미터 반환
            optimized_params = {
                'model_path': model_path,
                'params': best_params,
                'feature_importance': dict(zip(X.columns, best_model.feature_importances_))
            }
            
            self.logger.info(f"전략 파라미터 최적화 완료: {optimized_params}")
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"전략 파라미터 최적화 실패: {str(e)}")
            return None
            
    def analyze_trade_results(self) -> Dict:
        """
        거래 결과 분석
        
        Returns:
            Dict: 분석 결과
        """
        try:
            # 최근 거래 기록 조회
            trades = self.db_manager.get_trades(
                start_time=datetime.now() - timedelta(days=30)
            )
            
            if len(trades) < 10:
                self.logger.warning("분석할 거래 데이터가 부족합니다.")
                return None
                
            # 거래 결과 분석
            analysis = {
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if t['pnl'] > 0]),
                'losing_trades': len([t for t in trades if t['pnl'] <= 0]),
                'total_pnl': sum(t['pnl'] for t in trades),
                'avg_pnl': np.mean([t['pnl'] for t in trades]),
                'max_drawdown': self._calculate_max_drawdown(trades),
                'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades),
                'profit_factor': self._calculate_profit_factor(trades),
                'avg_holding_time': np.mean([
                    (t['exit_time'] - t['entry_time']).total_seconds()
                    for t in trades
                ])
            }
            
            # 시장 상황별 성과 분석
            market_analysis = self._analyze_market_conditions(trades)
            analysis.update(market_analysis)
            
            self.logger.info(f"거래 결과 분석 완료: {analysis}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"거래 결과 분석 실패: {str(e)}")
            return None
            
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """
        최대 낙폭 계산
        
        Args:
            trades (List[Dict]): 거래 기록
            
        Returns:
            float: 최대 낙폭
        """
        try:
            cumulative_pnl = np.cumsum([t['pnl'] for t in trades])
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            return np.max(drawdown) if len(drawdown) > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"최대 낙폭 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """
        수익률 계산
        
        Args:
            trades (List[Dict]): 거래 기록
            
        Returns:
            float: 수익률
        """
        try:
            profits = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            losses = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            return profits / losses if losses > 0 else float('inf')
            
        except Exception as e:
            self.logger.error(f"수익률 계산 실패: {str(e)}")
            return 0.0
            
    def _analyze_market_conditions(self, trades: List[Dict]) -> Dict:
        """
        시장 상황별 성과 분석
        
        Args:
            trades (List[Dict]): 거래 기록
            
        Returns:
            Dict: 시장 상황별 분석 결과
        """
        try:
            market_analysis = {
                'volatility_performance': {},
                'trend_performance': {},
                'time_performance': {}
            }
            
            for trade in trades:
                # 시장 데이터 조회
                market_data = self.db_manager.get_market_data(
                    symbol=trade['symbol'],
                    start_time=trade['entry_time'],
                    end_time=trade['exit_time']
                )
                
                if market_data is not None:
                    # 변동성 구간별 성과
                    volatility = market_data['close'].std()
                    vol_range = 'high' if volatility > 0.02 else 'low'
                    if vol_range not in market_analysis['volatility_performance']:
                        market_analysis['volatility_performance'][vol_range] = []
                    market_analysis['volatility_performance'][vol_range].append(trade['pnl'])
                    
                    # 추세 구간별 성과
                    trend = 'up' if market_data['close'].iloc[-1] > market_data['close'].iloc[0] else 'down'
                    if trend not in market_analysis['trend_performance']:
                        market_analysis['trend_performance'][trend] = []
                    market_analysis['trend_performance'][trend].append(trade['pnl'])
                    
                    # 시간대별 성과
                    hour = trade['entry_time'].hour
                    time_range = 'day' if 9 <= hour < 17 else 'night'
                    if time_range not in market_analysis['time_performance']:
                        market_analysis['time_performance'][time_range] = []
                    market_analysis['time_performance'][time_range].append(trade['pnl'])
            
            # 평균 성과 계산
            for category in market_analysis:
                for condition in market_analysis[category]:
                    if market_analysis[category][condition]:
                        market_analysis[category][condition] = np.mean(market_analysis[category][condition])
            
            return market_analysis
            
        except Exception as e:
            self.logger.error(f"시장 상황별 성과 분석 실패: {str(e)}")
            return {} 