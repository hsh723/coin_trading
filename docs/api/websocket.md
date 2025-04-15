# WebSocket 시스템 API 문서

## 1. 개요
WebSocket 시스템은 실시간 양방향 통신을 위한 WebSocket 서버를 관리합니다.

## 2. 초기화
```python
from src.websocket.websocket_manager import WebSocketManager

ws = WebSocketManager(
    config_dir="./config",
    data_dir="./data"
)
```

## 3. 메서드

### 3.1 서버 시작/중지
```python
# WebSocket 서버 시작
ws.start()

# WebSocket 서버 중지
ws.stop()
```

### 3.2 연결 관리
```python
# 연결 수락
await ws.handle_connection(websocket, path)

# 연결 종료
await ws.close_connection(websocket)
```

### 3.3 메시지 처리
```python
# 메시지 수신
await ws.process_message(websocket, message)

# 메시지 전송
await ws.send_message(websocket, message)

# 브로드캐스트
await ws.broadcast(message)
```

### 3.4 이벤트 핸들러
```python
# 이벤트 핸들러 등록
ws.register_handler(event_type, handler)

# 이벤트 핸들러 해제
ws.unregister_handler(event_type)
```

## 4. 통계

### 4.1 통계 조회
```python
# 활성 연결 수
connections = ws.get_connection_count()

# 전송된 메시지 수
sent = ws.get_sent_count()

# 수신된 메시지 수
received = ws.get_received_count()

# 전체 통계
stats = ws.get_stats()
```

### 4.2 통계 초기화
```python
# 통계 초기화
ws.reset_stats()
```

## 5. 설정

### 5.1 기본 설정
```json
{
    "host": "localhost",
    "port": 8765,
    "ping_interval": 30,
    "ping_timeout": 10
}
```

## 6. 에러 처리
- 모든 메서드는 예외를 발생시킬 수 있습니다.
- 에러는 로그에 기록됩니다.
- 통계에 에러 수가 기록됩니다.

## 7. 주의사항
1. WebSocket 서버 시작 전에 설정 파일이 있어야 합니다.
2. 메시지는 JSON으로 직렬화됩니다.
3. 핑/퐁 메커니즘으로 연결 상태를 확인합니다. 