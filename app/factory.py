"""Wire the assistant together from settings (one place to change providers)."""

from app.assistant import Assistant
from app.config import Settings
from app.providers.gemini import GeminiProvider


def build_assistant(settings: Settings) -> Assistant:
    provider = GeminiProvider(settings.gemini_api_key, settings.main_model)
    return Assistant(provider, settings)
