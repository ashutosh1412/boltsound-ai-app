"""Wire the assistant together from settings (one place to change providers)."""

from app.assistant import Assistant
from app.cache import AnswerCache
from app.config import Settings
from app.gateway import ModelGateway
from app.handoff import HandoffQueue
from app.providers.gemini import GeminiProvider
from app.providers.openai_provider import OpenAIProvider
from app.retrieval import Retriever, load_chunks
from app.returns import ReturnDesk
from app.tools import OrderStore


def build_assistant(settings: Settings) -> Assistant:
    providers = [
        GeminiProvider(
            settings.gemini_api_key, settings.main_model, settings.timeout_s, settings.thinking_level
        )
    ]
    if settings.openai_api_key:  # a second company, so one outage can't take us down
        providers.append(OpenAIProvider(settings.openai_api_key, settings.fallback_model, settings.timeout_s))
    provider = ModelGateway(providers)
    retriever = Retriever(settings.gemini_api_key, load_chunks())
    router = GeminiProvider(settings.gemini_api_key, settings.small_model, settings.timeout_s)
    return Assistant(
        provider,
        settings,
        retriever,
        OrderStore(),
        router,
        HandoffQueue(),
        AnswerCache(judge=router),
        ReturnDesk(),
    )
