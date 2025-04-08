@echo off
cd /d %~dp0
start /B python src/main.py --mode paper
echo 트레이딩 봇이 백그라운드에서 실행 중입니다.
echo 종료하려면 작업 관리자에서 Python 프로세스를 종료하세요. 