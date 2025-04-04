"""
순환 참조 문제 해결 스크립트
"""

import os

# 모든 __init__.py 파일 내용 비우기
init_files = [
    'src/utils/__init__.py',
    'src/exchange/__init__.py',
    'src/bot/__init__.py',
    'src/analysis/__init__.py',
    'src/strategy/__init__.py',
    'src/database/__init__.py',
    'src/risk/__init__.py',
    'src/__init__.py'
]

for file_path in init_files:
    if os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('# Empty file to avoid circular imports\n')
        print(f"Emptied {file_path}")
    else:
        print(f"File not found: {file_path}")

print("Done! All __init__.py files have been emptied.") 