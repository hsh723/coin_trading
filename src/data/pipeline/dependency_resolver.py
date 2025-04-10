from typing import Dict, List
import networkx as nx
from dataclasses import dataclass

@dataclass
class DependencyResolution:
    execution_order: List[str]
    dependencies: Dict[str, List[str]]
    cycle_detected: bool
    warnings: List[str]

class DependencyResolver:
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        
    async def resolve_dependencies(self, task_map: Dict) -> DependencyResolution:
        """데이터 처리 의존성 해결"""
        for task_id, task_info in task_map.items():
            deps = task_info.get('dependencies', [])
            self.dependency_graph.add_node(task_id)
            for dep in deps:
                self.dependency_graph.add_edge(dep, task_id)
                
        try:
            execution_order = list(nx.topological_sort(self.dependency_graph))
            return DependencyResolution(
                execution_order=execution_order,
                dependencies={node: list(self.dependency_graph.predecessors(node)) 
                            for node in execution_order},
                cycle_detected=False,
                warnings=[]
            )
        except nx.NetworkXUnfeasible:
            return self._handle_dependency_cycle()
