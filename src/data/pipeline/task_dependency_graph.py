from typing import Dict, List
import networkx as nx
from dataclasses import dataclass

@dataclass
class GraphAnalysis:
    cycles: List[List[str]]
    critical_path: List[str]
    max_depth: int
    parallel_groups: List[List[str]]

class DependencyGraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    async def build_dependency_graph(self, tasks: Dict[str, List[str]]) -> GraphAnalysis:
        """작업 의존성 그래프 구축"""
        for task_id, dependencies in tasks.items():
            self.graph.add_node(task_id)
            for dep in dependencies:
                self.graph.add_edge(dep, task_id)
                
        return GraphAnalysis(
            cycles=list(nx.simple_cycles(self.graph)),
            critical_path=self._find_critical_path(),
            max_depth=self._calculate_max_depth(),
            parallel_groups=self._identify_parallel_groups()
        )
