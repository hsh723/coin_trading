import pytest
import sys
import os
from datetime import datetime
import json
import coverage

def run_tests():
    """모든 테스트를 실행하고 결과를 보고서로 생성"""
    # 커버리지 측정 시작
    cov = coverage.Coverage()
    cov.start()
    
    # 테스트 실행
    test_results = pytest.main([
        'tests/',
        '-v',
        '--junitxml=test_results.xml',
        '--cov=src',
        '--cov-report=term-missing',
        '--cov-report=html:coverage_report'
    ])
    
    # 커버리지 측정 종료 및 보고서 생성
    cov.stop()
    cov.save()
    cov.report()
    
    # 테스트 결과 요약 생성
    generate_summary(test_results)
    
    return test_results

def generate_summary(test_results):
    """테스트 결과 요약 생성"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'skipped_tests': 0,
        'error_tests': 0,
        'coverage': 0,
        'test_cases': []
    }
    
    # JUnit XML 결과 파일 파싱
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse('test_results.xml')
        root = tree.getroot()
        
        for testcase in root.findall('.//testcase'):
            test_case = {
                'name': testcase.get('name'),
                'class': testcase.get('classname'),
                'time': float(testcase.get('time')),
                'status': 'passed'
            }
            
            if testcase.find('failure') is not None:
                test_case['status'] = 'failed'
                summary['failed_tests'] += 1
            elif testcase.find('error') is not None:
                test_case['status'] = 'error'
                summary['error_tests'] += 1
            elif testcase.find('skipped') is not None:
                test_case['status'] = 'skipped'
                summary['skipped_tests'] += 1
            else:
                summary['passed_tests'] += 1
            
            summary['test_cases'].append(test_case)
            summary['total_tests'] += 1
        
        # 커버리지 정보 추가
        cov = coverage.Coverage()
        cov.load()
        summary['coverage'] = cov.report()
        
    except Exception as e:
        print(f"Error generating summary: {e}")
    
    # 요약을 JSON 파일로 저장
    with open('test_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # 콘솔에 요약 출력
    print("\nTest Summary:")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Errors: {summary['error_tests']}")
    print(f"Skipped: {summary['skipped_tests']}")
    print(f"Coverage: {summary['coverage']:.2f}%")

def main():
    """메인 함수"""
    print("Starting tests...")
    
    # 테스트 실행
    test_results = run_tests()
    
    # 종료 코드 설정
    sys.exit(test_results)

if __name__ == '__main__':
    pytest.main(["--cov=src", "--cov-report=html"])