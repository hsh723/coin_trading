# 메시지 큐 시스템 API 문서

## 1. 개요
메시지 큐 시스템은 RabbitMQ, Redis, ZeroMQ를 지원하는 메시지 큐 관리 시스템입니다.

## 2. 초기화
```python
from src.mq.message_queue import MessageQueue

mq = MessageQueue(
    config_dir="./config",
    data_dir="./data"
)
```

## 3. 메서드

### 3.1 시작/중지
```python
# 메시지 큐 시작
mq.start()

# 메시지 큐 중지
mq.stop()
```

### 3.2 메시지 발행
```python
# 메시지 발행
mq.publish(
    queue="my_queue",
    message="my_message",
    mq_type="rabbitmq"  # "rabbitmq", "redis", "zeromq"
)
```

### 3.3 메시지 구독
```python
# 메시지 구독
def callback(message):
    print(f"Received: {message}")

mq.subscribe(
    queue="my_queue",
    callback=callback,
    mq_type="rabbitmq"  # "rabbitmq", "redis", "zeromq"
)
```

### 3.4 구독 해제
```python
# 구독 해제
mq.unsubscribe(
    queue="my_queue",
    mq_type="rabbitmq"  # "rabbitmq", "redis", "zeromq"
)
```

## 4. 통계

### 4.1 통계 조회
```python
# 발행된 메시지 수
published = mq.get_published_count()

# 소비된 메시지 수
consumed = mq.get_consumed_count()

# 구독자 수
subscribers = mq.get_subscriber_count()

# 전체 통계
stats = mq.get_stats()
```

### 4.2 통계 초기화
```python
# 통계 초기화
mq.reset_stats()
```

## 5. 설정

### 5.1 기본 설정
```json
{
    "default": {
        "type": "rabbitmq",
        "host": "localhost",
        "port": 5672,
        "username": "guest",
        "password": "guest",
        "vhost": "/"
    }
}
```

## 6. 에러 처리
- 모든 메서드는 예외를 발생시킬 수 있습니다.
- 에러는 로그에 기록됩니다.
- 통계에 에러 수가 기록됩니다.

## 7. 주의사항
1. 메시지 큐 시작 전에 설정 파일이 있어야 합니다.
2. 메시지는 JSON으로 직렬화됩니다.
3. 구독 콜백은 비동기로 실행됩니다. 