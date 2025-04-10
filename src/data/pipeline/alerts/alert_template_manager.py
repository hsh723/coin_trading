from typing import Dict, List
from dataclasses import dataclass
import jinja2

@dataclass
class AlertTemplate:
    template_id: str
    content: str
    variables: List[str]
    format: str

class AlertTemplateManager:
    def __init__(self):
        self.templates = {}
        self.jinja_env = jinja2.Environment(
            autoescape=True,
            trim_blocks=True
        )
        
    async def render_template(self, template_id: str, data: Dict) -> str:
        """알림 템플릿 렌더링"""
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")
            
        template = self.templates[template_id]
        jinja_template = self.jinja_env.from_string(template.content)
        
        return jinja_template.render(**data)
