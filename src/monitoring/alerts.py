import telegram
from typing import Dict, List
import logging

class AlertSystem:
    def __init__(self, config: Dict):
        self.telegram_bot = telegram.Bot(token=config['telegram_token'])
        self.chat_id = config['telegram_chat_id']
        self.alert_rules = config.get('alert_rules', {})
        
    async def send_alert(self, message: str, level: str = 'info') -> None:
        """알림 전송"""
        try:
            await self.telegram_bot.send_message(
                chat_id=self.chat_id,
                text=f"[{level.upper()}] {message}"
            )
        except Exception as e:
            logging.error(f"Failed to send alert: {str(e)}")

    def check_conditions(self, data: Dict) -> List[str]:
        """알림 조건 확인"""
        triggered_alerts = []
        for rule_name, rule in self.alert_rules.items():
            if self._evaluate_condition(data, rule['condition']):
                triggered_alerts.append(rule['message'])
        return triggered_alerts
