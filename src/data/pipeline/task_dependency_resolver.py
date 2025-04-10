from typing import Dict, List
import networkx as nx
from dataclasses import dataclass

@dataclass
class DependencyResolution:
    execution_order: List[str]
    dependencies: Dict[str, List[str]]
    has_cycles: bool

class TaskDependencyResolver:
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        
    async def resolve_dependencies(self, tasks: Dict[str, List[str]]) -> DependencyResolution:
        """작업 의존성 해결 및 실행 순서 결정"""
        # 의존성 그래프 구축
        for task_id, deps in tasks.items():
            self.dependency_graph.add_node(task_id)
            for dep in deps:
                self.dependency_graph.add_edge(dep, task_id)
                
        try:
            execution_order = list(nx.topological_sort(self.dependency_graph))
            return DependencyResolution(
                execution_order=execution_order,
                dependencies=self._get_dependencies(),
                has_cycles=False
            )
        except nx.NetworkXUnfeasible:
            return self._handle_cyclic_dependencies()
