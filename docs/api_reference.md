# API 레퍼런스

## 데이터 수집기 (DataCollector)

### 초기화

```python
from src.data.collector import DataCollector

collector = DataCollector(
    exchange='binance',
    symbol='BTC/USDT',
    timeframe='1h'
)
```

### 메서드

#### `fetch_historical_data()`

과거 데이터를 가져옵니다.

```python
data = collector.fetch_historical_data(
    start_time='2023-01-01',
    end_time='2023-12-31'
)
```

**반환값:**
```python
{
    'timestamp': pd.DatetimeIndex,
    'open': pd.Series,
    'high': pd.Series,
    'low': pd.Series,
    'close': pd.Series,
    'volume': pd.Series
}
```

#### `fetch_realtime_data()`

실시간 데이터를 가져옵니다.

```python
for data in collector.fetch_realtime_data():
    print(data)
```

## 데이터 프로세서 (DataProcessor)

### 초기화

```python
from src.data.processor import DataProcessor

processor = DataProcessor()
```

### 메서드

#### `process_data()`

데이터를 처리합니다.

```python
processed_data = processor.process_data(raw_data)
```

**반환값:**
```python
{
    'timestamp': pd.DatetimeIndex,
    'open': pd.Series,
    'high': pd.Series,
    'low': pd.Series,
    'close': pd.Series,
    'volume': pd.Series,
    'rsi': pd.Series,
    'macd': pd.Series,
    'macd_signal': pd.Series,
    'macd_hist': pd.Series,
    'bb_upper': pd.Series,
    'bb_middle': pd.Series,
    'bb_lower': pd.Series
}
```

## 전략 (Strategy)

### 초기화

```python
from src.strategies import STRATEGIES

strategy = STRATEGIES['Momentum'](
    rsi_period=14,
    rsi_upper=70,
    rsi_lower=30
)
```

### 메서드

#### `generate_signals()`

매매 신호를 생성합니다.

```python
signal = strategy.generate_signals(data)
```

**반환값:**
- `1`: 매수 신호
- `-1`: 매도 신호
- `0`: 중립

#### `calculate_position_size()`

포지션 크기를 계산합니다.

```python
size = strategy.calculate_position_size(data)
```

**반환값:**
- `float`: 0.0 ~ 1.0 사이의 포지션 크기

#### `calculate_stop_loss()`

손절가를 계산합니다.

```python
stop_loss = strategy.calculate_stop_loss(data)
```

**반환값:**
- `float`: 손절가

#### `calculate_take_profit()`

익절가를 계산합니다.

```python
take_profit = strategy.calculate_take_profit(data)
```

**반환값:**
- `float`: 익절가

## 주문 실행기 (OrderExecutor)

### 초기화

```python
from src.execution.executor import OrderExecutor

executor = OrderExecutor(
    exchange='binance',
    api_key='your_api_key',
    api_secret='your_api_secret'
)
```

### 메서드

#### `execute_order()`

주문을 실행합니다.

```python
order = executor.execute_order(
    symbol='BTC/USDT',
    side='buy',
    amount=0.1,
    price=50000,
    stop_loss=49000,
    take_profit=52000
)
```

**반환값:**
```python
{
    'id': str,
    'symbol': str,
    'side': str,
    'amount': float,
    'price': float,
    'status': str,
    'stop_loss': float,
    'take_profit': float
}
```

#### `get_balance()`

계정 잔고를 조회합니다.

```python
balance = executor.get_balance('USDT')
```

**반환값:**
- `float`: 잔고

#### `get_positions()`

현재 포지션을 조회합니다.

```python
positions = executor.get_positions()
```

**반환값:**
```python
[
    {
        'symbol': str,
        'side': str,
        'amount': float,
        'entry_price': float,
        'current_price': float,
        'pnl': float
    }
]
```

## 알림기 (Notifier)

### 초기화

```python
from src.notification.notifier import Notifier

notifier = Notifier(
    telegram_token='your_telegram_token',
    telegram_chat_id='your_chat_id',
    email_sender='your_email@example.com',
    email_password='your_email_password',
    email_recipient='recipient@example.com'
)
```

### 메서드

#### `send_telegram_message()`

텔레그램 메시지를 전송합니다.

```python
notifier.send_telegram_message('매수 신호 발생!')
```

#### `send_email()`

이메일을 전송합니다.

```python
notifier.send_email(
    subject='매수 신호',
    body='BTC/USDT 매수 신호가 발생했습니다.'
)
```

## 유틸리티 함수

### `calculate_rsi()`

RSI를 계산합니다.

```python
from src.utils.indicators import calculate_rsi

rsi = calculate_rsi(prices, period=14)
```

### `calculate_macd()`

MACD를 계산합니다.

```python
from src.utils.indicators import calculate_macd

macd, signal, hist = calculate_macd(
    prices,
    fast_period=12,
    slow_period=26,
    signal_period=9
)
```

### `calculate_bollinger_bands()`

볼린저 밴드를 계산합니다.

```python
from src.utils.indicators import calculate_bollinger_bands

upper, middle, lower = calculate_bollinger_bands(
    prices,
    period=20,
    std_dev=2
)
```

## 다음 단계

- [사용 가이드](usage.md)를 참조하여 시스템 사용법을 확인하세요.
- [문제 해결 가이드](troubleshooting.md)를 참조하여 문제를 해결하세요. 