"""
Sphinx 설정 파일
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# 프로젝트 정보
project = '암호화폐 트레이딩 시스템'
copyright = '2024, Your Name'
author = 'Your Name'

# Sphinx 확장
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
]

# 템플릿 경로
templates_path = ['_templates']

# 정적 파일 경로
html_static_path = ['_static']

# 기본 도메인
primary_domain = 'py'

# 기본 역할
default_role = 'py:obj'

# 자동 문서화 설정
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon 설정
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# HTML 출력 설정
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# LaTeX 출력 설정
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'figure_align': 'htbp',
}

# PDF 출력 설정
pdf_documents = [
    ('index', u'암호화폐트레이딩시스템', u'암호화폐 트레이딩 시스템 문서', u'Your Name'),
] 