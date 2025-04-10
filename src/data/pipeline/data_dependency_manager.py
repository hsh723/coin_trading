from typing import Dict, List
import networkx as nx
from dataclasses import dataclass

@dataclass
class DependencyNode:
    node_id: str
    dependencies: List[str]
    data_type: str
    update_frequency: str

class DataDependencyManager:
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        
    async def manage_dependencies(self, data_updates: Dict[str, any]) -> List[str]:
        """데이터 의존성 관리"""
        affected_nodes = []
        for node_id, update in data_updates.items():
            if node_id in self.dependency_graph:
                descendants = nx.descendants(self.dependency_graph, node_id)
                affected_nodes.extend(descendants)
                await self._propagate_updates(node_id, descendants)
        
        return list(set(affected_nodes))
