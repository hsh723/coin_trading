# 문제 해결 가이드

## 일반적인 문제

### 1. 가상환경 활성화 실패

**증상:**
```
-bash: source: No such file or directory
```

**해결 방법:**
1. 가상환경이 생성되었는지 확인:
```bash
ls -la venv/
```

2. 가상환경이 없다면 다시 생성:
```bash
python3 -m venv venv
```

3. 가상환경 활성화:
```bash
source venv/bin/activate
```

### 2. 패키지 설치 실패

**증상:**
```
ERROR: Could not find a version that satisfies the requirement
```

**해결 방법:**
1. pip 업그레이드:
```bash
pip install --upgrade pip
```

2. 패키지 캐시 삭제:
```bash
pip cache purge
```

3. 패키지 재설치:
```bash
pip install -r requirements.txt
```

### 3. 권한 문제

**증상:**
```
Permission denied
```

**해결 방법:**
1. 스크립트 실행 권한 부여:
```bash
chmod +x scripts/*.sh
```

2. 디렉토리 권한 확인:
```bash
ls -la
```

3. 필요한 경우 권한 수정:
```bash
chmod -R 755 .
```

## 데이터 수집 문제

### 1. API 연결 실패

**증상:**
```
Connection refused
```

**해결 방법:**
1. API 키 확인:
```python
print(settings['api']['key'])
```

2. 인터넷 연결 확인:
```bash
ping binance.com
```

3. 방화벽 설정 확인:
```bash
sudo ufw status
```

### 2. 데이터 누락

**증상:**
```
Missing data points
```

**해결 방법:**
1. 데이터 수집기 재시작:
```bash
./scripts/run_trader.sh
```

2. 로그 확인:
```bash
tail -f logs/trader.log
```

3. 데이터 저장소 확인:
```bash
ls -la data/raw/
```

## 전략 실행 문제

### 1. 신호 생성 실패

**증상:**
```
Error generating signals
```

**해결 방법:**
1. 전략 파라미터 확인:
```python
print(strategy.__dict__)
```

2. 데이터 형식 확인:
```python
print(data.head())
```

3. 로그 확인:
```bash
tail -f logs/strategy.log
```

### 2. 백테스팅 오류

**증상:**
```
Backtesting failed
```

**해결 방법:**
1. 데이터 기간 확인:
```python
print(data.index[0], data.index[-1])
```

2. 메모리 사용량 확인:
```python
import psutil
print(psutil.virtual_memory())
```

3. 백테스팅 파라미터 조정:
```python
strategy.backtest(data, start_date='2023-01-01', end_date='2023-12-31')
```

## 주문 실행 문제

### 1. 주문 실패

**증상:**
```
Order execution failed
```

**해결 방법:**
1. 잔고 확인:
```python
balance = executor.get_balance('USDT')
print(f"USDT Balance: {balance}")
```

2. 주문 파라미터 확인:
```python
print(order_params)
```

3. 거래소 상태 확인:
```python
status = exchange.fetch_status()
print(status)
```

### 2. 포지션 관리 오류

**증상:**
```
Position management error
```

**해결 방법:**
1. 포지션 상태 확인:
```python
positions = executor.get_positions()
print(positions)
```

2. 손절/익절 설정 확인:
```python
print(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")
```

3. 포지션 정리:
```python
executor.close_all_positions()
```

## 알림 문제

### 1. 텔레그램 알림 실패

**증상:**
```
Telegram notification failed
```

**해결 방법:**
1. 텔레그램 토큰 확인:
```python
print(settings['notification']['telegram']['token'])
```

2. 채팅 ID 확인:
```python
print(settings['notification']['telegram']['chat_id'])
```

3. 텔레그램 봇 상태 확인:
```python
bot = telegram.Bot(token=token)
print(bot.get_me())
```

### 2. 이메일 알림 실패

**증상:**
```
Email notification failed
```

**해결 방법:**
1. 이메일 설정 확인:
```python
print(settings['notification']['email'])
```

2. SMTP 서버 연결 테스트:
```python
import smtplib
smtp = smtplib.SMTP('smtp.gmail.com', 587)
smtp.starttls()
smtp.login(email, password)
smtp.quit()
```

3. 이메일 템플릿 확인:
```python
print(email_template)
```

## 대시보드 문제

### 1. 대시보드 접속 실패

**증상:**
```
Cannot connect to dashboard
```

**해결 방법:**
1. 포트 확인:
```bash
netstat -tulpn | grep 8501
```

2. 방화벽 설정 확인:
```bash
sudo ufw status
```

3. 대시보드 재시작:
```bash
./scripts/run_dashboard.sh
```

### 2. 데이터 표시 오류

**증상:**
```
Data display error
```

**해결 방법:**
1. 데이터 로드 확인:
```python
print(data.head())
```

2. 차트 설정 확인:
```python
print(chart_config)
```

3. 캐시 삭제:
```python
cache.clear()
```

## 다음 단계

- [설치 가이드](installation.md)를 참조하여 시스템을 재설치하세요.
- [사용 가이드](usage.md)를 참조하여 시스템 사용법을 확인하세요. 