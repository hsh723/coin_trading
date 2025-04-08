from web3 import Web3
from typing import Dict
import asyncio

class DexInterface:
    def __init__(self, network_config: Dict):
        self.w3 = Web3(Web3.HTTPProvider(network_config['rpc_url']))
        self.router_address = network_config['router_address']
        self.router_abi = network_config['router_abi']
        self.router_contract = self.w3.eth.contract(
            address=self.router_address,
            abi=self.router_abi
        )
        
    async def get_token_price(self, token_address: str, base_token_address: str) -> float:
        """토큰 가격 조회"""
        amounts_out = await self.router_contract.functions.getAmountsOut(
            1e18,  # 1 token
            [token_address, base_token_address]
        ).call()
        return amounts_out[1] / 1e18
