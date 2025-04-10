from typing import List, Dict
from dataclasses import dataclass
import openai

@dataclass
class Content:
    title: str
    script: str
    thumbnail_text: List[str]
    midjourney_prompt: str
    tags: List[str]

class ContentGenerator:
    def __init__(self, channel_config: Dict):
        self.config = channel_config
        self.templates = {
            'korea_facts': {
                'hook': "충격적인 사실을 아십니까? {topic}...",
                'structure': ["충격적 사실", "역사적 맥락", "자부심 고조", "현재 위상"]
            },
            'discover_korea': {
                'hook': "Have you ever wondered why Koreans {topic}?",
                'structure': ["cultural shock", "historical context", "modern twist"]
            },
            'health_info': {
                'hook': "이것만 알면 건강해진다! {topic}의 놀라운 효과",
                'structure': ["건강 효과", "실천 방법", "전문가 조언"]
            }
        }
        
    async def generate_content(self, topic: str) -> Content:
        """주제별 콘텐츠 생성"""
        template = self.templates[self.config['channel_id']]
        
        script = await self._generate_script(topic, template)
        thumbnail_texts = await self._generate_thumbnail_texts(topic)
        
        return Content(
            title=self._generate_title(topic),
            script=script,
            thumbnail_text=thumbnail_texts,
            midjourney_prompt=self._generate_midjourney_prompt(topic),
            tags=self._generate_tags(topic)
        )
