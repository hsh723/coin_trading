# 전략 개발 가이드

## 전략 개발 개요

### 1. 전략 구조

모든 거래 전략은 `src/strategies/` 디렉토리에 구현되어야 하며, 다음 구조를 따릅니다:

```python
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data):
        """매매 신호 생성"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, data):
        """포지션 크기 계산"""
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, data):
        """손절가 계산"""
        pass
    
    @abstractmethod
    def calculate_take_profit(self, data):
        """익절가 계산"""
        pass
```

### 2. 전략 등록

새로운 전략을 등록하려면 `src/strategies/__init__.py`에 추가하세요:

```python
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy

STRATEGIES = {
    'Momentum': MomentumStrategy,
    'Mean Reversion': MeanReversionStrategy,
    'Breakout': BreakoutStrategy
}
```

## 기본 전략 구현

### 1. 모멘텀 전략

```python
class MomentumStrategy(BaseStrategy):
    def __init__(self, rsi_period=14, rsi_upper=70, rsi_lower=30):
        self.rsi_period = rsi_period
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
    
    def generate_signals(self, data):
        # RSI 계산
        rsi = calculate_rsi(data['close'], self.rsi_period)
        
        # 매수 신호
        if rsi < self.rsi_lower:
            return 1
        
        # 매도 신호
        if rsi > self.rsi_upper:
            return -1
        
        return 0
    
    def calculate_position_size(self, data):
        # 기본 포지션 크기
        return 1.0
    
    def calculate_stop_loss(self, data):
        # ATR 기반 손절가
        atr = calculate_atr(data, 14)
        return data['close'].iloc[-1] - (2 * atr)
    
    def calculate_take_profit(self, data):
        # ATR 기반 익절가
        atr = calculate_atr(data, 14)
        return data['close'].iloc[-1] + (4 * atr)
```

### 2. 평균 회귀 전략

```python
class MeanReversionStrategy(BaseStrategy):
    def __init__(self, ma_period=20, std_dev=2):
        self.ma_period = ma_period
        self.std_dev = std_dev
    
    def generate_signals(self, data):
        # 이동평균 및 표준편차 계산
        ma = data['close'].rolling(self.ma_period).mean()
        std = data['close'].rolling(self.ma_period).std()
        
        # 현재 가격
        current_price = data['close'].iloc[-1]
        
        # 매수 신호 (가격이 평균보다 낮을 때)
        if current_price < ma.iloc[-1] - (std.iloc[-1] * self.std_dev):
            return 1
        
        # 매도 신호 (가격이 평균보다 높을 때)
        if current_price > ma.iloc[-1] + (std.iloc[-1] * self.std_dev):
            return -1
        
        return 0
```

### 3. 브레이크아웃 전략

```python
class BreakoutStrategy(BaseStrategy):
    def __init__(self, period=20, atr_multiplier=2):
        self.period = period
        self.atr_multiplier = atr_multiplier
    
    def generate_signals(self, data):
        # 고가/저가 채널 계산
        high_channel = data['high'].rolling(self.period).max()
        low_channel = data['low'].rolling(self.period).min()
        
        # 현재 가격
        current_price = data['close'].iloc[-1]
        
        # 상단 브레이크아웃
        if current_price > high_channel.iloc[-1]:
            return 1
        
        # 하단 브레이크아웃
        if current_price < low_channel.iloc[-1]:
            return -1
        
        return 0
```

## 전략 최적화

### 1. 파라미터 최적화

```python
def optimize_parameters(strategy_class, data, param_grid):
    best_params = None
    best_sharpe = -float('inf')
    
    for params in param_grid:
        strategy = strategy_class(**params)
        returns = backtest(strategy, data)
        sharpe = calculate_sharpe_ratio(returns)
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
    
    return best_params, best_sharpe
```

### 2. 성과 평가

```python
def evaluate_strategy(strategy, data):
    returns = backtest(strategy, data)
    
    metrics = {
        'total_return': calculate_total_return(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns)
    }
    
    return metrics
```

## 전략 테스트

### 1. 백테스팅

```python
def backtest(strategy, data):
    positions = []
    returns = []
    
    for i in range(len(data)):
        signal = strategy.generate_signals(data.iloc[:i+1])
        
        if signal == 1 and not positions:  # 매수
            entry_price = data['close'].iloc[i]
            positions.append(('long', entry_price))
        
        elif signal == -1 and positions:  # 매도
            exit_price = data['close'].iloc[i]
            position = positions.pop()
            returns.append((exit_price - position[1]) / position[1])
    
    return pd.Series(returns)
```

### 2. 포워드 테스팅

```python
def forward_test(strategy, data, live_data):
    # 과거 데이터로 전략 초기화
    strategy.initialize(data)
    
    # 실시간 데이터로 테스트
    for new_data in live_data:
        signal = strategy.generate_signals(new_data)
        if signal != 0:
            execute_trade(signal, new_data)
```

## 리스크 관리

### 1. 포지션 사이징

```python
def calculate_position_size(data, risk_per_trade=0.02):
    account_value = get_account_value()
    stop_loss = calculate_stop_loss(data)
    risk_amount = account_value * risk_per_trade
    
    position_size = risk_amount / (data['close'].iloc[-1] - stop_loss)
    return min(position_size, 1.0)  # 최대 100% 제한
```

### 2. 손절/익절 관리

```python
def manage_trade(position, current_price, stop_loss, take_profit):
    if current_price <= stop_loss:
        return 'close', 'stop_loss'
    elif current_price >= take_profit:
        return 'close', 'take_profit'
    return 'hold', None
```

## 다음 단계

- [API 레퍼런스](api_reference.md)를 참조하여 API 사용법을 확인하세요.
- [문제 해결 가이드](troubleshooting.md)를 참조하여 문제를 해결하세요. 