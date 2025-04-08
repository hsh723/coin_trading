"""
API 문서 자동 생성 스크립트
"""

import os
import re
import ast
from typing import List, Dict
import markdown
from datetime import datetime

def extract_docstring(node) -> str:
    """
    노드에서 독스트링 추출
    
    Args:
        node: AST 노드
        
    Returns:
        str: 독스트링
    """
    if not node.body or not isinstance(node.body[0], ast.Expr):
        return ""
        
    docstring = ast.get_docstring(node)
    return docstring if docstring else ""

def parse_function_docstring(docstring: str) -> Dict:
    """
    함수 독스트링 파싱
    
    Args:
        docstring (str): 독스트링
        
    Returns:
        Dict: 파싱된 정보
    """
    if not docstring:
        return {}
        
    sections = {
        'description': '',
        'args': [],
        'returns': '',
        'raises': []
    }
    
    current_section = 'description'
    lines = docstring.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('Args:'):
            current_section = 'args'
            continue
        elif line.startswith('Returns:'):
            current_section = 'returns'
            continue
        elif line.startswith('Raises:'):
            current_section = 'raises'
            continue
            
        if current_section == 'description':
            sections['description'] += line + '\n'
        elif current_section == 'args':
            if line and not line.startswith('    '):
                match = re.match(r'(\w+)\s*\(.*?\):\s*(.*)', line)
                if match:
                    sections['args'].append({
                        'name': match.group(1),
                        'description': match.group(2)
                    })
        elif current_section == 'returns':
            sections['returns'] += line + '\n'
        elif current_section == 'raises':
            if line and not line.startswith('    '):
                match = re.match(r'(\w+):\s*(.*)', line)
                if match:
                    sections['raises'].append({
                        'type': match.group(1),
                        'description': match.group(2)
                    })
                    
    return sections

def generate_markdown(module_path: str) -> str:
    """
    마크다운 문서 생성
    
    Args:
        module_path (str): 모듈 경로
        
    Returns:
        str: 마크다운 문서
    """
    with open(module_path, 'r', encoding='utf-8') as f:
        source = f.read()
        
    tree = ast.parse(source)
    
    markdown_content = []
    module_docstring = extract_docstring(tree)
    
    if module_docstring:
        markdown_content.append(f"# {os.path.basename(module_path)}\n")
        markdown_content.append(module_docstring.strip() + "\n")
        
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_docstring = extract_docstring(node)
            if class_docstring:
                markdown_content.append(f"## {node.name}\n")
                markdown_content.append(class_docstring.strip() + "\n")
                
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    func_docstring = extract_docstring(item)
                    if func_docstring:
                        markdown_content.append(f"### {item.name}\n")
                        doc_info = parse_function_docstring(func_docstring)
                        
                        if doc_info['description']:
                            markdown_content.append(doc_info['description'].strip() + "\n")
                            
                        if doc_info['args']:
                            markdown_content.append("#### 매개변수\n")
                            markdown_content.append("| 이름 | 설명 |\n")
                            markdown_content.append("|------|------|\n")
                            for arg in doc_info['args']:
                                markdown_content.append(f"| {arg['name']} | {arg['description']} |\n")
                                
                        if doc_info['returns']:
                            markdown_content.append("#### 반환값\n")
                            markdown_content.append(doc_info['returns'].strip() + "\n")
                            
                        if doc_info['raises']:
                            markdown_content.append("#### 예외\n")
                            markdown_content.append("| 타입 | 설명 |\n")
                            markdown_content.append("|------|------|\n")
                            for exc in doc_info['raises']:
                                markdown_content.append(f"| {exc['type']} | {exc['description']} |\n")
                                
    return "\n".join(markdown_content)

def generate_api_docs():
    """API 문서 생성"""
    # 소스 디렉토리 설정
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
    
    # 문서 디렉토리 생성
    os.makedirs(docs_dir, exist_ok=True)
    
    # 모듈 목록
    modules = [
        'indicators/technical_analyzer.py',
        'database/database_manager.py',
        'utils/security_manager.py',
        'utils/monitoring_dashboard.py',
        'utils/performance_reporter.py',
        'utils/feedback_system.py'
    ]
    
    # 각 모듈에 대한 문서 생성
    for module in modules:
        module_path = os.path.join(src_dir, module)
        if os.path.exists(module_path):
            markdown_content = generate_markdown(module_path)
            
            # 마크다운 파일 저장
            doc_path = os.path.join(docs_dir, f"{os.path.splitext(module)[0]}.md")
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
                
    # 인덱스 페이지 생성
    index_content = [
        "# API 문서",
        f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 모듈 목록",
        ""
    ]
    
    for module in modules:
        doc_path = f"{os.path.splitext(module)[0]}.md"
        module_name = os.path.basename(module)
        index_content.append(f"- [{module_name}]({doc_path})")
        
    with open(os.path.join(docs_dir, 'index.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(index_content))
        
if __name__ == '__main__':
    generate_api_docs() 