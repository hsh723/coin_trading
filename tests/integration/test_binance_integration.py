import pytest
from src.exchange.binance_client import BinanceClient
from src.config.env_loader import EnvLoader
import logging
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def client():
    """Binance 클라이언트 픽스처"""
    client = BinanceClient(
        api_key="test_key",
        api_secret="test_secret",
        test_mode=True
    )
    client.initialize()
    return client

def test_market_data(client):
    """시장 데이터 조회 테스트"""
    symbol = "BTCUSDT"
    data = client.get_market_data(symbol)
    assert data is not None
    assert "price" in data
    assert "bid" in data
    assert "ask" in data
    assert "volume" in data
    assert "price_change" in data

def test_order_creation(client):
    """주문 생성 테스트"""
    symbol = "BTCUSDT"
    side = "BUY"
    type = "MARKET"
    quantity = 0.001
    
    order = client.create_order(
        symbol=symbol,
        side=side,
        type=type,
        quantity=quantity
    )
    assert order is not None
    assert "orderId" in order
    assert order["symbol"] == symbol
    assert order["side"] == side
    assert order["type"] == type
    assert float(order["origQty"]) == quantity

def test_account_balance(client):
    """계정 잔고 조회 테스트"""
    balances = client.get_account_balance()
    assert balances is not None
    assert isinstance(balances, dict)
    assert "BTC" in balances
    assert "free" in balances["BTC"]
    assert "locked" in balances["BTC"]

def test_open_orders(client):
    """미체결 주문 조회 테스트"""
    symbol = "BTCUSDT"
    orders = client.get_open_orders(symbol)
    assert orders is not None
    assert isinstance(orders, list)
    for order in orders:
        assert order["symbol"] == symbol
        assert order["status"] == "NEW" 