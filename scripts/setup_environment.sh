#!/bin/bash

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")/.."

# 로그 디렉토리 생성
mkdir -p logs

# 로그 파일 설정
LOG_FILE="logs/setup_$(date +%Y%m%d_%H%M%S).log"

# 로깅 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 에러 처리 함수
handle_error() {
    log "오류 발생: $1"
    exit 1
}

# Python 버전 확인
if ! command -v python3 &> /dev/null; then
    handle_error "Python3가 설치되어 있지 않습니다."
fi

# 가상환경 생성
log "Python 가상환경 생성..."
python3 -m venv venv || handle_error "가상환경 생성 실패"

# 가상환경 활성화
source venv/bin/activate || handle_error "가상환경 활성화 실패"

# pip 업그레이드
log "pip 업그레이드..."
pip install --upgrade pip || handle_error "pip 업그레이드 실패"

# 필요한 패키지 설치
log "필요한 패키지 설치..."
pip install -r requirements.txt || handle_error "패키지 설치 실패"

# 설정 디렉토리 생성
log "설정 디렉토리 생성..."
mkdir -p config/backup || handle_error "설정 디렉토리 생성 실패"

# 기본 설정 파일 생성
if [ ! -f "config/settings.json" ]; then
    log "기본 설정 파일 생성..."
    cat > config/settings.json << EOL
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
    "email_enabled": false,
    "last_updated": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOL
fi

# 데이터 디렉토리 생성
log "데이터 디렉토리 생성..."
mkdir -p data/{raw,processed,backup} || handle_error "데이터 디렉토리 생성 실패"

# 스크립트 실행 권한 설정
log "스크립트 실행 권한 설정..."
chmod +x scripts/*.sh || handle_error "스크립트 권한 설정 실패"

log "환경 설정이 완료되었습니다."
log "거래 시스템을 시작하려면 ./scripts/run_trader.sh를 실행하세요."
log "대시보드를 시작하려면 ./scripts/run_dashboard.sh를 실행하세요." 