from typing import Dict, List
from dataclasses import dataclass
import jinja2

@dataclass
class Template:
    template_id: str
    content: str
    variables: List[str]
    format_type: str

class AlertTemplating:
    def __init__(self):
        self.templates = {}
        self.environment = jinja2.Environment(
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
    async def render_alert(self, template_id: str, data: Dict) -> str:
        """알림 템플릿 렌더링"""
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")
            
        template = self.environment.from_string(
            self.templates[template_id].content
        )
        
        return template.render(**data)
