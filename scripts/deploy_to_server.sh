#!/bin/bash

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")/.."

# 로그 디렉토리 생성
mkdir -p logs

# 로그 파일 설정
LOG_FILE="logs/deploy_$(date +%Y%m%d_%H%M%S).log"

# 로깅 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 에러 처리 함수
handle_error() {
    log "오류 발생: $1"
    exit 1
}

# 서버 정보 설정
if [ -z "$1" ]; then
    handle_error "서버 주소를 입력해주세요. (예: ./deploy_to_server.sh user@server.com)"
fi

SERVER=$1
REMOTE_DIR="/opt/crypto_trader"

# SSH 키 확인
if [ ! -f ~/.ssh/id_rsa ]; then
    log "SSH 키가 없습니다. 생성합니다..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" || handle_error "SSH 키 생성 실패"
fi

# 서버에 SSH 키 복사
log "서버에 SSH 키 복사..."
ssh-copy-id $SERVER || handle_error "SSH 키 복사 실패"

# 서버에 필요한 디렉토리 생성
log "서버에 디렉토리 생성..."
ssh $SERVER "sudo mkdir -p $REMOTE_DIR && sudo chown -R \$USER:$USER $REMOTE_DIR" || handle_error "서버 디렉토리 생성 실패"

# 프로젝트 파일 복사
log "프로젝트 파일 복사..."
rsync -avz --exclude 'venv' \
    --exclude 'data/backup' \
    --exclude 'logs' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    ./ $SERVER:$REMOTE_DIR/ || handle_error "파일 복사 실패"

# 서버에서 환경 설정 실행
log "서버에서 환경 설정 실행..."
ssh $SERVER "cd $REMOTE_DIR && ./scripts/setup_environment.sh" || handle_error "서버 환경 설정 실패"

# 서비스 파일 생성
log "시스템 서비스 파일 생성..."
cat > crypto_trader.service << EOL
[Unit]
Description=Crypto Trading System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REMOTE_DIR
ExecStart=$REMOTE_DIR/scripts/run_trader.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

# 서비스 파일 복사 및 활성화
log "서비스 설정..."
scp crypto_trader.service $SERVER:/tmp/ || handle_error "서비스 파일 복사 실패"
ssh $SERVER "sudo mv /tmp/crypto_trader.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable crypto_trader && sudo systemctl start crypto_trader" || handle_error "서비스 설정 실패"

# 대시보드 서비스 파일 생성
log "대시보드 서비스 파일 생성..."
cat > crypto_dashboard.service << EOL
[Unit]
Description=Crypto Trading Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REMOTE_DIR
ExecStart=$REMOTE_DIR/scripts/run_dashboard.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

# 대시보드 서비스 파일 복사 및 활성화
log "대시보드 서비스 설정..."
scp crypto_dashboard.service $SERVER:/tmp/ || handle_error "대시보드 서비스 파일 복사 실패"
ssh $SERVER "sudo mv /tmp/crypto_dashboard.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable crypto_dashboard && sudo systemctl start crypto_dashboard" || handle_error "대시보드 서비스 설정 실패"

# 임시 파일 삭제
rm -f crypto_trader.service crypto_dashboard.service

# 배포 완료 메시지
log "배포가 완료되었습니다."
log "서비스 상태 확인:"
ssh $SERVER "sudo systemctl status crypto_trader && sudo systemctl status crypto_dashboard"
log "대시보드 접속: http://$SERVER:8501" 