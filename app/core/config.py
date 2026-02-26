"""HealthGuard — Configuration

All settings loaded from environment variables.
Venice AI: STT, Vision, TTS, ImgGen, Chat
AkashML: Structuring, reasoning, fallback
Telegram: Alert delivery
"""
import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class VeniceConfig(BaseModel):
    api_key: str = os.getenv("VENICE_API_KEY", "")
    base_url: str = os.getenv("VENICE_BASE_URL", "https://api.venice.ai/api/v1")
    chat_model: str = os.getenv("VENICE_CHAT_MODEL", "venice-uncensored")
    vision_model: str = os.getenv("VENICE_VISION_MODEL", "qwen3-vl-235b-a22b")
    image_model: str = os.getenv("VENICE_IMAGE_MODEL", "fluently-xl")
    audio_model: str = os.getenv("VENICE_AUDIO_MODEL", "tts-kokoro")
    stt_model: str = os.getenv("VENICE_STT_MODEL", "whisper-large-v3")
    embedding_model: str = os.getenv("VENICE_EMBEDDING_MODEL", "text-embedding-ada-002")


class AkashMLConfig(BaseModel):
    api_key: str = os.getenv("AKASHML_API_KEY", "")
    base_url: str = os.getenv("AKASHML_BASE_URL", "https://api.akashml.com/v1")
    primary_model: str = os.getenv("AKASHML_PRIMARY_MODEL", "Meta-Llama-3-3-70B-Instruct")
    heavy_model: str = os.getenv("AKASHML_HEAVY_MODEL", "deepseek-ai/DeepSeek-V3.1")
    fallback_model: str = os.getenv("AKASHML_FALLBACK_MODEL", "Meta-Llama-3-3-70B-Instruct")


class TelegramConfig(BaseModel):
    bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")


class AppConfig(BaseModel):
    venice: VeniceConfig = VeniceConfig()
    akashml: AkashMLConfig = AkashMLConfig()
    telegram: TelegramConfig = TelegramConfig()
    data_dir: str = os.getenv("DATA_DIR", "/data")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))
    agent_interval: int = int(os.getenv("AGENT_INTERVAL", "60"))
    raw_file_ttl: int = int(os.getenv("RAW_FILE_TTL", "60"))
    encryption_salt: str = os.getenv("ENCRYPTION_SALT", "healthguard_default_salt_change_me")
    demo_mode: bool = os.getenv("DEMO_MODE", "true").lower() == "true"


def get_config() -> AppConfig:
    return AppConfig()
