"""Wire the assistant together from settings (one place to change providers)."""

from pathlib import Path

from app.assistant import Assistant
from app.cache import AnswerCache
from app.config import Settings
from app.feedback import FeedbackLog
from app.gateway import ModelGateway
from app.handoff import HandoffQueue
from app.monitoring import RequestLog
from app.providers.gemini import GeminiProvider
from app.providers.openai_provider import OpenAIProvider
from app.retrieval import Retriever, load_chunks
from app.returns import ReturnDesk
from app.tools import OrderStore

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def build_assistant(settings: Settings, log_dir: Path = LOG_DIR) -> Assistant:
    providers = [
        GeminiProvider(
            settings.gemini_api_key, settings.main_model, settings.timeout_s, settings.thinking_level
        )
    ]
    if settings.openai_api_key:  # a second company, so one outage can't take us down
        providers.append(OpenAIProvider(settings.openai_api_key, settings.fallback_model, settings.timeout_s))
    router = GeminiProvider(settings.gemini_api_key, settings.small_model, settings.timeout_s)
    return Assistant(
        provider=ModelGateway(providers),
        router=router,
        retriever=Retriever(settings.gemini_api_key, load_chunks()),
        orders=OrderStore(),
        cache=AnswerCache(judge=router),
        returns=ReturnDesk(log_dir / "returns.jsonl"),
        handoffs=HandoffQueue(log_dir / "handoffs.jsonl"),
        monitor=RequestLog(log_dir / "requests.jsonl"),
        feedback=FeedbackLog(log_dir / "feedback.jsonl"),
    )
