"""
자산 캐시 관리자 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from src.execution.asset_cache_manager import AssetCacheManager

@pytest.fixture
def config():
    """테스트 설정"""
    return {
        'cache_ttl': 10,  # 10초
        'refresh_interval': 0.1,  # 100ms
        'max_cache_size': 100,
        'default_symbols': ['BTC/USDT', 'ETH/USDT'],
        'max_trades_per_symbol': 10
    }

@pytest.fixture
async def cache_manager(config):
    """캐시 관리자 인스턴스"""
    manager = AssetCacheManager(config)
    await manager.initialize()
    yield manager
    await manager.close()

@pytest.mark.asyncio
async def test_initialization(config):
    """초기화 테스트"""
    manager = AssetCacheManager(config)
    await manager.initialize()
    
    # 설정 확인
    assert manager.cache_ttl == 10
    assert manager.refresh_interval == 0.1
    assert manager.max_cache_size == 100
    assert len(manager.subscribed_symbols) == 2
    assert 'BTC/USDT' in manager.subscribed_symbols
    assert 'ETH/USDT' in manager.subscribed_symbols
    assert manager.is_running == True
    assert len(manager.tasks) == 2
    
    await manager.close()

@pytest.mark.asyncio
async def test_subscribe_unsubscribe(cache_manager):
    """구독 및 구독 해제 테스트"""
    # 초기 구독 확인
    assert len(cache_manager.subscribed_symbols) == 2
    
    # 구독 추가
    await cache_manager.subscribe_symbol('XRP/USDT')
    assert 'XRP/USDT' in cache_manager.subscribed_symbols
    assert len(cache_manager.subscribed_symbols) == 3
    
    # 중복 구독 시도
    await cache_manager.subscribe_symbol('XRP/USDT')
    assert len(cache_manager.subscribed_symbols) == 3  # 변경 없음
    
    # 구독 해제
    await cache_manager.unsubscribe_symbol('XRP/USDT')
    assert 'XRP/USDT' not in cache_manager.subscribed_symbols
    assert len(cache_manager.subscribed_symbols) == 2
    
    # 미구독 심볼 해제 시도
    await cache_manager.unsubscribe_symbol('LTC/USDT')
    assert len(cache_manager.subscribed_symbols) == 2  # 변경 없음

@pytest.mark.asyncio
async def test_price_data(cache_manager):
    """가격 데이터 테스트"""
    # 초기 데이터 확인
    btc_price = cache_manager.get_price('BTC/USDT')
    assert btc_price is not None
    
    # 유효성 확인
    assert cache_manager.is_price_valid('BTC/USDT') == True
    assert cache_manager.is_price_valid('LTC/USDT') == False
    
    # 모든 가격 조회
    all_prices = cache_manager.get_all_prices()
    assert len(all_prices) >= 2
    assert 'BTC/USDT' in all_prices
    assert 'ETH/USDT' in all_prices

@pytest.mark.asyncio
async def test_orderbook_data(cache_manager):
    """오더북 데이터 테스트"""
    # 초기 데이터 확인
    orderbook = cache_manager.get_orderbook('BTC/USDT')
    assert orderbook is not None
    assert 'bids' in orderbook
    assert 'asks' in orderbook
    assert len(orderbook['bids']) > 0
    assert len(orderbook['asks']) > 0

@pytest.mark.asyncio
async def test_trade_data(cache_manager):
    """체결 내역 테스트"""
    # 초기 데이터 확인
    initial_trades = cache_manager.get_recent_trades('BTC/USDT')
    assert isinstance(initial_trades, list)
    
    # 체결 내역 추가
    trade1 = {
        'id': '1',
        'price': 50000.0,
        'amount': 1.0,
        'side': 'buy'
    }
    cache_manager.add_trade('BTC/USDT', trade1)
    
    trade2 = {
        'id': '2',
        'price': 50100.0,
        'amount': 0.5,
        'side': 'sell'
    }
    cache_manager.add_trade('BTC/USDT', trade2)
    
    # 추가된 내역 확인
    updated_trades = cache_manager.get_recent_trades('BTC/USDT')
    assert len(updated_trades) >= 2
    assert updated_trades[-1]['id'] == '2'
    assert updated_trades[-2]['id'] == '1'
    
    # 한도 지정 조회
    limited_trades = cache_manager.get_recent_trades('BTC/USDT', limit=1)
    assert len(limited_trades) == 1
    assert limited_trades[0]['id'] == '1'  # 최신 순 정렬

@pytest.mark.asyncio
async def test_asset_info(cache_manager):
    """자산 정보 테스트"""
    # 초기 데이터 확인
    initial_info = cache_manager.get_asset_info('BTC/USDT')
    
    # 자산 정보 업데이트
    asset_info = {
        'base_asset': 'BTC',
        'quote_asset': 'USDT',
        'min_qty': 0.0001,
        'max_qty': 1000.0,
        'price_precision': 2,
        'qty_precision': 6
    }
    cache_manager.update_asset_info('BTC/USDT', asset_info)
    
    # 업데이트된 정보 확인
    updated_info = cache_manager.get_asset_info('BTC/USDT')
    assert updated_info is not None
    assert updated_info['base_asset'] == 'BTC'
    assert updated_info['quote_asset'] == 'USDT'
    assert 'timestamp' in updated_info

@pytest.mark.asyncio
async def test_market_data(cache_manager):
    """시장 데이터 테스트"""
    # 시장 데이터 업데이트
    market_data = {
        'volume': 1000.0,
        'change': 2.5,
        'high': 51000.0,
        'low': 49000.0
    }
    cache_manager.update_market_data('BTC/USDT', market_data)
    
    # 업데이트된 데이터 확인 (기능 추가 필요)
    # 현재 구현에서는 직접 조회 방법이 없음

@pytest.mark.asyncio
async def test_cache_expiry(cache_manager, monkeypatch):
    """캐시 만료 테스트"""
    # 현재 시간 조작을 위한 몽키패치
    original_now = datetime.now
    
    try:
        # 체결 내역 추가
        now = original_now()
        
        for i in range(5):
            trade = {
                'id': str(i),
                'price': 50000.0 + i * 100,
                'amount': 1.0,
                'timestamp': now
            }
            cache_manager.add_trade('BTC/USDT', trade)
        
        # 초기 내역 확인
        initial_trades = cache_manager.get_recent_trades('BTC/USDT')
        assert len(initial_trades) == 5
        
        # 시간 조작 (캐시 만료 시간 이후)
        future_time = now + timedelta(seconds=cache_manager.cache_ttl + 1)
        
        def mock_now():
            return future_time
            
        monkeypatch.setattr(datetime, 'now', mock_now)
        
        # 만료된 캐시 정리
        cache_manager._remove_expired_cache()
        
        # 정리 후 내역 확인
        updated_trades = cache_manager.get_recent_trades('BTC/USDT')
        assert len(updated_trades) == 0
        
    finally:
        # 원래 시간 함수 복원
        monkeypatch.setattr(datetime, 'now', original_now)

@pytest.mark.asyncio
async def test_cache_pruning(cache_manager):
    """캐시 정리 테스트"""
    # 미구독 심볼 데이터 추가
    for symbol in ['LTC/USDT', 'XRP/USDT', 'DOT/USDT']:
        price_data = {
            'price': 100.0,
            'timestamp': datetime.now() - timedelta(seconds=cache_manager.cache_ttl * 3)
        }
        cache_manager.price_cache[symbol] = price_data
        cache_manager.last_update_time[symbol] = datetime.now() - timedelta(seconds=cache_manager.cache_ttl * 3)
    
    # 초기 캐시 확인
    assert 'LTC/USDT' in cache_manager.price_cache
    assert 'XRP/USDT' in cache_manager.price_cache
    assert 'DOT/USDT' in cache_manager.price_cache
    
    # 캐시 정리 실행
    await cache_manager._prune_cache()
    
    # 정리 후 확인
    assert 'LTC/USDT' not in cache_manager.price_cache
    assert 'XRP/USDT' not in cache_manager.price_cache
    assert 'DOT/USDT' not in cache_manager.price_cache

@pytest.mark.asyncio
async def test_cache_size(cache_manager):
    """캐시 크기 계산 테스트"""
    # 초기 크기 확인
    initial_size = cache_manager._get_cache_size()
    
    # 데이터 추가
    for i in range(10):
        symbol = f"TEST{i}/USDT"
        price_data = {
            'price': 100.0 + i,
            'timestamp': datetime.now()
        }
        cache_manager.price_cache[symbol] = price_data
        
        orderbook_data = {
            'bids': [(99.0 + i, 1.0)],
            'asks': [(101.0 + i, 1.0)],
            'timestamp': datetime.now()
        }
        cache_manager.orderbook_cache[symbol] = orderbook_data
        
        trade = {
            'id': str(i),
            'price': 100.0 + i,
            'amount': 1.0,
            'timestamp': datetime.now()
        }
        cache_manager.add_trade(symbol, trade)
    
    # 증가된 크기 확인
    updated_size = cache_manager._get_cache_size()
    assert updated_size > initial_size
    assert updated_size >= initial_size + 30  # 각 심볼당 3개 항목 추가

@pytest.mark.asyncio
async def test_cleanup(cache_manager):
    """정리 테스트"""
    # 데이터 추가
    symbol = "TEST/USDT"
    price_data = {
        'price': 100.0,
        'timestamp': datetime.now()
    }
    cache_manager.price_cache[symbol] = price_data
    
    # 정리 실행
    await cache_manager.close()
    
    # 상태 확인
    assert cache_manager.is_running == False
    assert len(cache_manager.price_cache) == 0
    assert len(cache_manager.orderbook_cache) == 0
    assert len(cache_manager.trade_cache) == 0 