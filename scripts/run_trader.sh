#!/bin/bash

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")/.."

# 로그 디렉토리 생성
mkdir -p logs

# 로그 파일 설정
LOG_FILE="logs/trader_$(date +%Y%m%d_%H%M%S).log"

# 로깅 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 에러 처리 함수
handle_error() {
    log "오류 발생: $1"
    exit 1
}

# Python 가상환경 확인
if [ ! -d "venv" ]; then
    log "Python 가상환경이 없습니다. setup_environment.sh를 실행해주세요."
    exit 1
fi

# 가상환경 활성화
source venv/bin/activate || handle_error "가상환경 활성화 실패"

# 필요한 패키지 설치 확인
pip list | grep -q "ccxt" || handle_error "ccxt 패키지가 설치되어 있지 않습니다."
pip list | grep -q "pandas" || handle_error "pandas 패키지가 설치되어 있지 않습니다."
pip list | grep -q "numpy" || handle_error "numpy 패키지가 설치되어 있지 않습니다."

# 설정 파일 확인
if [ ! -f "config/settings.json" ]; then
    log "설정 파일이 없습니다. config/settings.json을 생성해주세요."
    exit 1
fi

# 거래 시스템 실행
log "거래 시스템 시작..."
python src/main.py >> "$LOG_FILE" 2>&1 &

# 프로세스 ID 저장
echo $! > .trader.pid

# 실행 상태 확인
sleep 5
if ps -p $(cat .trader.pid) > /dev/null; then
    log "거래 시스템이 성공적으로 시작되었습니다. (PID: $(cat .trader.pid))"
else
    handle_error "거래 시스템 시작 실패"
fi

# 로그 모니터링
tail -f "$LOG_FILE" 