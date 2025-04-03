#!/bin/bash

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")/.."

# 로그 디렉토리 생성
mkdir -p logs

# 로그 파일 설정
LOG_FILE="logs/backup_$(date +%Y%m%d_%H%M%S).log"

# 로깅 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 에러 처리 함수
handle_error() {
    log "오류 발생: $1"
    exit 1
}

# 백업 디렉토리 설정
BACKUP_DIR="data/backup"
BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"

# 백업 디렉토리 생성
mkdir -p "$BACKUP_DIR/$BACKUP_NAME" || handle_error "백업 디렉토리 생성 실패"

# 데이터 백업
log "데이터 백업 시작..."

# 설정 파일 백업
if [ -f "config/settings.json" ]; then
    log "설정 파일 백업..."
    cp config/settings.json "$BACKUP_DIR/$BACKUP_NAME/" || handle_error "설정 파일 백업 실패"
fi

# 거래 데이터 백업
if [ -d "data/raw" ]; then
    log "거래 데이터 백업..."
    cp -r data/raw/* "$BACKUP_DIR/$BACKUP_NAME/raw/" || handle_error "거래 데이터 백업 실패"
fi

# 처리된 데이터 백업
if [ -d "data/processed" ]; then
    log "처리된 데이터 백업..."
    cp -r data/processed/* "$BACKUP_DIR/$BACKUP_NAME/processed/" || handle_error "처리된 데이터 백업 실패"
fi

# 로그 파일 백업
if [ -d "logs" ]; then
    log "로그 파일 백업..."
    cp -r logs/* "$BACKUP_DIR/$BACKUP_NAME/logs/" || handle_error "로그 파일 백업 실패"
fi

# 백업 압축
log "백업 파일 압축..."
tar -czf "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" -C "$BACKUP_DIR" "$BACKUP_NAME" || handle_error "백업 압축 실패"

# 임시 디렉토리 삭제
rm -rf "$BACKUP_DIR/$BACKUP_NAME" || handle_error "임시 디렉토리 삭제 실패"

# 오래된 백업 삭제 (30일 이상)
log "오래된 백업 정리..."
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +30 -delete || handle_error "오래된 백업 삭제 실패"

# 백업 완료 메시지
BACKUP_SIZE=$(du -h "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" | cut -f1)
log "백업이 완료되었습니다. (크기: $BACKUP_SIZE)"
log "백업 파일: $BACKUP_DIR/${BACKUP_NAME}.tar.gz" 