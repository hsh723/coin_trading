import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_config() -> Dict[str, Any]:
    """
    설정 파일 로드
    
    Returns:
        Dict[str, Any]: 설정 데이터
    """
    try:
        # 기본 설정
        default_config = {
            'database': {
                'path': 'data/trading.db',
                'backup_dir': 'data/backups',
                'backup_interval': 24,
                'max_backups': 7,
                'encryption_key': 'your-secret-key-here'
            },
            'exchange': {
                'name': 'binance',
                'testnet': True,
                'api_key': 'your-api-key',
                'api_secret': 'your-api-secret'
            },
            'trading': {
                'initial_capital': 10000,
                'position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'max_positions': 3,
                'min_volume': 1000000,
                'max_spread': 0.002
            },
            'backtest': {
                'initial_capital': 10000,
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'timeframes': ['1h', '4h', '1d']
            },
            'simulation': {
                'speed': 1.0,
                'initial_capital': 10000
            },
            'telegram': {
                'bot_token': 'your-bot-token',
                'chat_id': 'your-chat-id',
                'notifications': {
                    'trade': True,
                    'error': True,
                    'performance': True,
                    'news': True
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/trading.log',
                'max_size': 10485760,
                'backup_count': 5,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'performance_analysis': {
                'min_samples': 100,
                'training_interval': 24,
                'indicators': {
                    'ma_short': 5,
                    'ma_long': 20,
                    'rsi_period': 14,
                    'rsi_upper': 70,
                    'rsi_lower': 30,
                    'bb_period': 20,
                    'bb_std': 2
                },
                'thresholds': {
                    'volatility': 0.02,
                    'volume': 1000000,
                    'sentiment': -0.5
                }
            },
            'news_analysis': {
                'sources': ['coindesk', 'cointelegraph', 'bitcoinist'],
                'update_interval': 1,
                'sentiment_threshold': -0.5,
                'keywords': {
                    'positive': ['bullish', 'surge', 'rally', 'breakout'],
                    'negative': ['bearish', 'crash', 'dump', 'selloff'],
                    'categories': ['bitcoin', 'ethereum', 'regulation', 'technology', 'market']
                }
            }
        }
        
        # 설정 파일 경로
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
        
        # 설정 파일이 존재하는 경우 로드
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                
            # 사용자 설정으로 기본 설정 업데이트
            def update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        d[k] = update_dict(d[k], v)
                    else:
                        d[k] = v
                return d
            
            config = update_dict(default_config, user_config)
        else:
            logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
            config = default_config
        
        return config
        
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류 발생: {str(e)}")
        raise 