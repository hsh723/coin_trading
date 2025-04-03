#!/usr/bin/env python
"""
암호화폐 트레이딩 시스템 실행 스크립트
"""

import sys
import os

# 현재 디렉토리를 시스템 경로에 추가
sys.path.insert(0, os.path.abspath('.'))

# main.py에서 필요한 함수 import
from src.main import main

if __name__ == "__main__":
    # 커맨드 라인 인자 전달
    main(sys.argv[1:]) 