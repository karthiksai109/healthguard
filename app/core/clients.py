"""HealthGuard — API Clients

OpenAI-compatible clients for Venice AI and AkashML.
Telegram client for alert delivery.
"""
import httpx
from openai import OpenAI
from app.core.config import AppConfig


def get_venice_client(config: AppConfig) -> OpenAI:
    return OpenAI(
        api_key=config.venice.api_key,
        base_url=config.venice.base_url,
    )


def get_akashml_client(config: AppConfig) -> OpenAI:
    return OpenAI(
        api_key=config.akashml.api_key,
        base_url=config.akashml.base_url,
    )


class TelegramClient:
    """Sends alerts via Telegram Bot API with verifiable receipts."""

    def __init__(self, config: AppConfig):
        self.token = config.telegram.bot_token
        self.chat_id = config.telegram.chat_id
        self.enabled = bool(self.token and self.chat_id)

    def send_message(self, text: str) -> dict:
        if not self.enabled:
            return {"ok": False, "reason": "telegram_not_configured", "status_code": 0}
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            with httpx.Client(timeout=10.0) as http:
                resp = http.post(url, json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"})
                return {
                    "ok": resp.status_code == 200,
                    "status_code": resp.status_code,
                    "response": resp.text[:200],
                }
        except Exception as e:
            return {"ok": False, "status_code": 0, "error": str(e)}

    def send_audio(self, audio_bytes: bytes, caption: str = "") -> dict:
        if not self.enabled:
            return {"ok": False, "reason": "telegram_not_configured", "status_code": 0}
        url = f"https://api.telegram.org/bot{self.token}/sendVoice"
        try:
            with httpx.Client(timeout=15.0) as http:
                resp = http.post(
                    url,
                    data={"chat_id": self.chat_id, "caption": caption[:1024]},
                    files={"voice": ("alert.mp3", audio_bytes, "audio/mpeg")},
                )
                return {"ok": resp.status_code == 200, "status_code": resp.status_code}
        except Exception as e:
            return {"ok": False, "status_code": 0, "error": str(e)}
