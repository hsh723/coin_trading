#!/usr/bin/env python
"""
베이지안 시계열 예측기 실행 스크립트

다양한 베이지안 시계열 모델을 사용하여 암호화폐 가격 예측을 수행합니다.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# 상위 디렉토리 임포트를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.bayesian.examples.crypto_price_prediction import run_crypto_price_prediction
from src.ml.bayesian.examples.model_comparison import run_model_comparison
from src.ml.bayesian.examples.ensemble_prediction import run_ensemble_prediction, compare_ensemble_methods
from src.ml.bayesian.examples.online_learning_example import run_online_learning_example, compare_models_online
from src.ml.bayesian.model_factory import BayesianModelFactory

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"bayesian_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """
    명령줄 인수 파싱
    
    Returns:
        명령줄 인수
    """
    parser = argparse.ArgumentParser(description='베이지안 시계열 예측기')
    
    parser.add_argument('--mode', type=str, default='single',
                      choices=['single', 'compare', 'ensemble', 'online', 'all'],
                      help='실행 모드 (단일 모델, 모델 비교, 앙상블 모델, 온라인 학습, 모든 모델)')
    
    parser.add_argument('--model', type=str, default='ar',
                      choices=['ar', 'gp', 'structural'],
                      help='사용할 모델 유형')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                      help='암호화폐 심볼 (예: BTCUSDT, ETHUSDT)')
    
    parser.add_argument('--days', type=int, default=30,
                      help='예측할 미래 일수 또는 데이터 학습 일수')
    
    parser.add_argument('--ar_order', type=int, default=5,
                      help='AR 모델의 차수')
    
    parser.add_argument('--seasonality', action='store_true',
                      help='계절성 포함 여부')
    
    parser.add_argument('--season_period', type=int, default=7,
                      help='계절성 주기')
    
    parser.add_argument('--kernel', type=str, default='rbf',
                      choices=['rbf', 'matern32', 'matern52', 'exponential', 'periodic'],
                      help='GP 모델 커널 유형')
    
    parser.add_argument('--trend', action='store_true',
                      help='추세 포함 여부')
    
    parser.add_argument('--damped_trend', action='store_true',
                      help='감쇠 추세 사용 여부')
    
    parser.add_argument('--ensemble_method', type=str, default='weighted',
                      choices=['mean', 'weighted', 'median', 'bayes'],
                      help='앙상블 방법')
    
    parser.add_argument('--compare_ensembles', action='store_true',
                      help='다양한 앙상블 방법 비교 실행 여부')
    
    parser.add_argument('--show_individual', action='store_true',
                      help='앙상블 그래프에 개별 모델 예측 표시 여부')
    
    parser.add_argument('--timeframe', type=str, default='1d',
                      choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                      help='시간 프레임')
    
    parser.add_argument('--window_size', type=int, default=180,
                      help='온라인 학습을 위한 윈도우 크기')
    
    parser.add_argument('--update_freq', type=int, default=7,
                      help='온라인 학습 업데이트 주기')
    
    parser.add_argument('--compare_online', action='store_true',
                      help='온라인 학습에서 다양한 모델 비교 실행 여부')
    
    return parser.parse_args()

def get_model_params(args: argparse.Namespace, model_type: str) -> Dict[str, Any]:
    """
    인수에서 모델 파라미터 추출
    
    Args:
        args: 명령줄 인수
        model_type: 모델 유형
        
    Returns:
        모델 파라미터 딕셔너리
    """
    params = {}
    
    if model_type == 'ar':
        params = {
            'ar_order': args.ar_order,
            'seasonality': args.seasonality,
            'num_seasons': args.season_period
        }
    elif model_type == 'gp':
        params = {
            'kernel_type': args.kernel,
            'seasonality': args.seasonality,
            'period': args.season_period,
            'trend': args.trend
        }
    elif model_type == 'structural':
        params = {
            'level': True,
            'trend': args.trend,
            'seasonality': args.seasonality,
            'season_period': args.season_period,
            'damped_trend': args.damped_trend
        }
    
    return params

def main() -> None:
    """
    메인 실행 함수
    """
    args = parse_args()
    
    try:
        if args.mode == 'single':
            logger.info(f"단일 모델 실행: {args.model}")
            model_params = get_model_params(args, args.model)
            
            run_crypto_price_prediction(
                model_type=args.model,
                symbol=args.symbol,
                n_forecast=args.days,
                model_params=model_params
            )
            
        elif args.mode == 'compare':
            logger.info("모델 비교 실행")
            run_model_comparison(
                symbol=args.symbol,
                n_forecast=args.days
            )
            
        elif args.mode == 'ensemble':
            logger.info("앙상블 모델 실행")
            
            if args.compare_ensembles:
                compare_ensemble_methods(
                    symbol=args.symbol,
                    n_forecast=args.days
                )
            else:
                run_ensemble_prediction(
                    symbol=args.symbol,
                    n_forecast=args.days,
                    ensemble_method=args.ensemble_method,
                    show_individual=args.show_individual
                )
                
        elif args.mode == 'online':
            logger.info("온라인 학습 모델 실행")
            
            if args.compare_online:
                compare_models_online(
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    days=args.days
                )
            else:
                run_online_learning_example(
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    days=args.days,
                    model_type=args.model,
                    window_size=args.window_size,
                    update_freq=args.update_freq
                )
            
        elif args.mode == 'all':
            logger.info("모든 모델 순차 실행")
            
            # AR 모델
            logger.info("AR 모델 실행")
            ar_params = get_model_params(args, 'ar')
            run_crypto_price_prediction(
                model_type='ar',
                symbol=args.symbol,
                n_forecast=args.days,
                model_params=ar_params
            )
            
            # GP 모델
            logger.info("GP 모델 실행")
            gp_params = get_model_params(args, 'gp')
            run_crypto_price_prediction(
                model_type='gp',
                symbol=args.symbol,
                n_forecast=args.days,
                model_params=gp_params
            )
            
            # 구조적 시계열 모델
            logger.info("구조적 시계열 모델 실행")
            structural_params = get_model_params(args, 'structural')
            run_crypto_price_prediction(
                model_type='structural',
                symbol=args.symbol,
                n_forecast=args.days,
                model_params=structural_params
            )
            
            # 모델 비교
            logger.info("모델 비교 실행")
            run_model_comparison(
                symbol=args.symbol,
                n_forecast=args.days
            )
            
            # 앙상블 모델
            logger.info("앙상블 모델 실행")
            run_ensemble_prediction(
                symbol=args.symbol,
                n_forecast=args.days,
                ensemble_method=args.ensemble_method,
                show_individual=args.show_individual
            )
            
            # 온라인 학습 모델
            logger.info("온라인 학습 모델 실행")
            run_online_learning_example(
                symbol=args.symbol,
                timeframe=args.timeframe,
                days=min(args.days, 14),  # 온라인 학습은 데이터 양 제한
                model_type='ar',
                window_size=args.window_size,
                update_freq=args.update_freq
            )
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 