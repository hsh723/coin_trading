from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ChannelConfig:
    channel_id: str
    name: str
    target_audience: str
    language: str
    style_guide: Dict
    hook_structure: List[str]

class YoutubeChannelManager:
    def __init__(self):
        self.channels = {
            'korea_facts': ChannelConfig(
                channel_id='korea_facts',
                name='코리아팩트',
                target_audience='국내 20-40대',
                language='ko',
                style_guide={
                    'tone': 'emotional',
                    'thumbnail_style': 'dramatic',
                    'hook_duration': 10
                },
                hook_structure=[
                    'shocking_fact',
                    'emotional_connection',
                    'historical_pride'
                ]
            ),
            'discover_korea': ChannelConfig(
                channel_id='discover_korea',
                name='Discover Korea',
                target_audience='Global viewers',
                language='en',
                style_guide={
                    'tone': 'informative',
                    'thumbnail_style': 'cultural',
                    'hook_duration': 10
                },
                hook_structure=[
                    'cultural_surprise',
                    'global_comparison',
                    'modern_twist'
                ]
            ),
            'health_info': ChannelConfig(
                channel_id='health_info',
                name='건강정보TV',
                target_audience='50-70대',
                language='ko',
                style_guide={
                    'tone': 'friendly',
                    'thumbnail_style': 'clear',
                    'hook_duration': 10
                },
                hook_structure=[
                    'health_benefit',
                    'daily_tips',
                    'expert_advice'
                ]
            )
        }
