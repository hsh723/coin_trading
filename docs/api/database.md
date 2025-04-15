# 데이터베이스 시스템 API 문서

## 1. 개요
데이터베이스 시스템은 SQLite, PostgreSQL, MySQL을 지원하는 데이터베이스 관리 시스템입니다.

## 2. 초기화
```python
from src.database.db_manager import DatabaseManager

db = DatabaseManager(
    config_dir="./config",
    data_dir="./data"
)
```

## 3. 메서드

### 3.1 시작/중지
```python
# 데이터베이스 시작
db.start()

# 데이터베이스 중지
db.stop()
```

### 3.2 연결 관리
```python
# 연결 생성
connection = db.create_connection(
    db_type="sqlite",  # "sqlite", "postgresql", "mysql"
    config={
        "host": "localhost",
        "port": 5432,
        "username": "user",
        "password": "password",
        "database": "mydb"
    }
)

# 연결 종료
db.close_connection(connection)
```

### 3.3 쿼리 실행
```python
# 쿼리 실행
result = db.execute_query(
    db_type="sqlite",
    config={
        "host": "localhost",
        "port": 5432,
        "username": "user",
        "password": "password",
        "database": "mydb"
    },
    query="SELECT * FROM users",
    params=None
)
```

### 3.4 트랜잭션 관리
```python
# 트랜잭션 시작
db.begin_transaction(connection)

# 트랜잭션 커밋
db.commit_transaction(connection)

# 트랜잭션 롤백
db.rollback_transaction(connection)
```

## 4. 통계

### 4.1 통계 조회
```python
# 활성 연결 수
connections = db.get_connection_count()

# 실행된 쿼리 수
queries = db.get_query_count()

# 트랜잭션 수
transactions = db.get_transaction_count()

# 전체 통계
stats = db.get_stats()
```

### 4.2 통계 초기화
```python
# 통계 초기화
db.reset_stats()
```

## 5. 설정

### 5.1 기본 설정
```json
{
    "default": {
        "type": "sqlite",
        "host": "localhost",
        "port": 5432,
        "username": "user",
        "password": "password",
        "database": "mydb"
    }
}
```

## 6. 에러 처리
- 모든 메서드는 예외를 발생시킬 수 있습니다.
- 에러는 로그에 기록됩니다.
- 통계에 에러 수가 기록됩니다.

## 7. 주의사항
1. 데이터베이스 시작 전에 설정 파일이 있어야 합니다.
2. 연결은 자동으로 풀링됩니다.
3. 트랜잭션은 명시적으로 관리해야 합니다. 