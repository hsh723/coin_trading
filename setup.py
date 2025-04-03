from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="coin_trading",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="암호화폐 자동매매 시스템",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto_trader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ccxt>=4.1.13",
        "pandas>=2.1.4",
        "numpy>=1.26.2",
        "python-telegram-bot>=20.7",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "requests>=2.31.0",
        "websockets>=12.0",
        "aiohttp>=3.9.1",
        "schedule>=1.2.1",
        "structlog>=24.1.0",
        "prometheus-client>=0.19.0",
        "ta>=0.11.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "streamlit>=1.10.0",
        "backtrader>=1.9.76.123",
        "empyrical>=0.5.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.6",
            "ipykernel>=6.27.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-trader=src.main:main",
        ],
    },
) 