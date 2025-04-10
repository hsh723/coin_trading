from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ChainLink:
    task_id: str
    input_deps: List[str]
    output_deps: List[str]
    state: str

class TaskChainManager:
    def __init__(self):
        self.chains = {}
        self.active_links = set()
        
    async def process_chain(self, chain_config: Dict) -> List[ChainLink]:
        """작업 체인 처리"""
        chain_id = chain_config['id']
        links = []
        
        for task_config in chain_config['tasks']:
            link = ChainLink(
                task_id=task_config['id'],
                input_deps=task_config.get('inputs', []),
                output_deps=task_config.get('outputs', []),
                state='pending'
            )
            links.append(link)
            self.active_links.add(link.task_id)
            
        self.chains[chain_id] = links
        return await self._execute_chain(chain_id)
