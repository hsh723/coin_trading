# 설치 및 설정 가이드

## 시스템 요구사항

- Python 3.8 이상
- Linux/Unix 운영체제 (Ubuntu 20.04 LTS 권장)
- 최소 4GB RAM
- 20GB 이상의 저장 공간

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/yourusername/crypto_trader.git
cd crypto_trader
```

### 2. 환경 설정

환경 설정 스크립트를 실행하여 필요한 모든 의존성을 설치합니다:

```bash
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

이 스크립트는 다음 작업을 수행합니다:
- Python 가상환경 생성
- 필요한 패키지 설치
- 설정 디렉토리 생성
- 기본 설정 파일 생성
- 데이터 디렉토리 생성

### 3. 설정 파일 구성

기본 설정 파일은 `config/settings.json`에 생성됩니다. 다음과 같이 수정할 수 있습니다:

```json
{
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "strategy": "Momentum",
    "rsi_period": 14,
    "rsi_upper": 70,
    "position_size": 1.0,
    "stop_loss": 2.0,
    "take_profit": 4.0,
    "max_positions": 3,
    "telegram_enabled": false,
    "email_enabled": false
}
```

### 4. API 키 설정

거래소 API 키를 설정하려면:

1. 거래소 계정에서 API 키 생성
2. `config/api_keys.json` 파일 생성:

```json
{
    "binance": {
        "api_key": "your_api_key",
        "api_secret": "your_api_secret"
    }
}
```

## 서버 배포

### 1. 서버 준비

```bash
# 서버에 필요한 패키지 설치
sudo apt update
sudo apt install python3-venv python3-pip git

# 방화벽 설정
sudo ufw allow 22
sudo ufw allow 8501  # 대시보드 포트
```

### 2. 자동 배포

배포 스크립트를 사용하여 서버에 자동으로 배포할 수 있습니다:

```bash
./scripts/deploy_to_server.sh user@your-server.com
```

이 스크립트는 다음 작업을 수행합니다:
- SSH 키 설정
- 프로젝트 파일 복사
- 서비스 설정
- 자동 시작 구성

## 문제 해결

### 1. 가상환경 활성화 실패

```bash
# 가상환경 재생성
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

### 2. 패키지 설치 실패

```bash
# pip 업그레이드
pip install --upgrade pip

# 패키지 재설치
pip install -r requirements.txt
```

### 3. 권한 문제

```bash
# 스크립트 실행 권한 설정
chmod +x scripts/*.sh

# 데이터 디렉토리 권한 설정
chmod -R 755 data/
```

## 다음 단계

- [사용 방법 가이드](usage.md)를 참조하여 시스템 사용법을 익히세요.
- [전략 개발 가이드](strategies.md)를 참조하여 거래 전략을 개발하세요.
- [API 레퍼런스](api_reference.md)를 참조하여 API 사용법을 확인하세요. 