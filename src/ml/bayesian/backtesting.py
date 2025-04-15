import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import json
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class BayesianBacktester:
    """
    베이지안 시계열 모델 백테스팅 시스템
    
    주요 기능:
    - 모델 성능 평가
    - 전략 최적화
    - 리스크 분석
    - 성과 지표 계산
    """
    
    def __init__(self,
                 model_type: str = "ar",
                 model_params: Dict[str, Any] = None,
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001,
                 save_dir: str = "./backtesting_results"):
        """
        백테스팅 시스템 초기화
        
        Args:
            model_type: 모델 유형
            model_params: 모델 파라미터
            initial_capital: 초기 자본금
            transaction_cost: 거래 비용
            save_dir: 결과 저장 디렉토리
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.save_dir = save_dir
        
        # 결과 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 백테스팅 결과 저장용 변수
        self.positions = []
        self.portfolio_values = []
        self.returns = []
        self.trades = []
        self.predictions = []
        self.actuals = []
    
    def run(self,
            data: pd.DataFrame,
            train_size: float = 0.7,
            prediction_horizon: int = 1) -> Dict[str, Any]:
        """
        백테스팅 실행
        
        Args:
            data: 시계열 데이터
            train_size: 학습 데이터 비율
            prediction_horizon: 예측 기간
            
        Returns:
            백테스팅 결과
        """
        logger.info("백테스팅 시작...")
        
        # 데이터 분할
        train_size = int(len(data) * train_size)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # 모델 초기화
        from .model_factory import BayesianModelFactory
        model = BayesianModelFactory.get_model(
            model_type=self.model_type,
            **self.model_params
        )
        
        # 모델 학습
        model.fit(train_data)
        
        # 포트폴리오 초기화
        portfolio_value = self.initial_capital
        position = 0.0
        
        # 백테스팅 루프
        for i in range(len(test_data) - prediction_horizon):
            # 현재 데이터
            current_data = test_data.iloc[i:i+1]
            
            # 예측
            prediction = model.predict(current_data, horizon=prediction_horizon)
            actual = test_data.iloc[i+prediction_horizon]['price']
            
            # 거래 신호 생성
            signal = self._generate_signal(prediction, actual)
            
            # 포지션 조정
            new_position = self._adjust_position(position, signal)
            
            # 거래 실행
            if new_position != position:
                trade_cost = abs(new_position - position) * self.transaction_cost
                portfolio_value -= trade_cost
            
            # 포트폴리오 가치 업데이트
            portfolio_value *= (1 + (new_position * (actual / current_data['price'].iloc[0] - 1)))
            
            # 결과 저장
            self.positions.append(new_position)
            self.portfolio_values.append(portfolio_value)
            self.returns.append(portfolio_value / self.initial_capital - 1)
            self.trades.append({
                'timestamp': current_data.index[0],
                'position': new_position,
                'price': current_data['price'].iloc[0],
                'cost': trade_cost
            })
            self.predictions.append(prediction)
            self.actuals.append(actual)
            
            # 포지션 업데이트
            position = new_position
        
        # 성과 지표 계산
        results = self._calculate_performance_metrics()
        
        # 결과 저장
        self._save_results(results)
        
        return results
    
    def _generate_signal(self, prediction: float, actual: float) -> float:
        """
        거래 신호 생성
        
        Args:
            prediction: 예측값
            actual: 실제값
            
        Returns:
            거래 신호 (-1: 매도, 0: 홀드, 1: 매수)
        """
        if prediction > actual * 1.01:  # 1% 이상 상승 예측
            return 1.0
        elif prediction < actual * 0.99:  # 1% 이상 하락 예측
            return -1.0
        else:
            return 0.0
    
    def _adjust_position(self, current_position: float, signal: float) -> float:
        """
        포지션 조정
        
        Args:
            current_position: 현재 포지션
            signal: 거래 신호
            
        Returns:
            새로운 포지션
        """
        if signal == 0:
            return current_position
        else:
            return signal
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        성과 지표 계산
        
        Returns:
            성과 지표
        """
        returns = np.array(self.returns)
        portfolio_values = np.array(self.portfolio_values)
        
        # 총 수익률
        total_return = (portfolio_values[-1] / self.initial_capital) - 1
        
        # 연간 수익률
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # 샤프 비율
        excess_returns = returns - 0.02/252  # 무위험 수익률 가정
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
        
        # 최대 낙폭
        max_drawdown = (portfolio_values / np.maximum.accumulate(portfolio_values) - 1).min()
        
        # 승률
        winning_trades = len([t for t in self.trades if t['position'] * (t['price'] - self.trades[self.trades.index(t)-1]['price']) > 0])
        win_rate = winning_trades / len(self.trades) if self.trades else 0
        
        # 거래 횟수
        trade_count = len([t for t in self.trades if t['position'] != 0])
        
        # 평균 거래 비용
        avg_cost = np.mean([t['cost'] for t in self.trades])
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trade_count': trade_count,
            'avg_cost': avg_cost
        }
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        결과 저장
        
        Args:
            results: 백테스팅 결과
        """
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.save_dir, f"backtest_results_{timestamp}.json")
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # 거래 내역 저장
        trades_file = os.path.join(self.save_dir, f"trades_{timestamp}.csv")
        pd.DataFrame(self.trades).to_csv(trades_file, index=False)
        
        # 포트폴리오 가치 저장
        portfolio_file = os.path.join(self.save_dir, f"portfolio_{timestamp}.csv")
        pd.DataFrame({
            'timestamp': [t['timestamp'] for t in self.trades],
            'value': self.portfolio_values
        }).to_csv(portfolio_file, index=False) 