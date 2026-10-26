"""Model gateway: every model call goes through one place that retries and falls back."""

import logging
import time
from collections.abc import Callable

from app.providers.base import LLMProvider, ProviderError
from app.types import Completion

log = logging.getLogger(__name__)

# Worth retrying: rate limits, server errors, timeouts. Not worth it: 400/401/404 will fail again.
TRANSIENT = ("429", "500", "502", "503", "504", "deadline", "timed out", "timeout", "unavailable")


class ModelGateway:
    """Try each provider in order. Retry a failed call once (with a short wait), then move on."""

    def __init__(
        self,
        providers: list[LLMProvider],
        retries: int = 1,
        backoff_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.providers = providers
        self.retries = retries
        self.backoff_s = backoff_s
        self._sleep = sleep
        self.name = " → ".join(p.name for p in providers)

    def complete(self, system: str, prompt: str) -> Completion:
        failures: list[str] = []
        for provider in self.providers:
            for attempt in range(self.retries + 1):
                try:
                    completion = provider.complete(system, prompt)
                except ProviderError as exc:
                    log.warning("model call failed (attempt %d): %s", attempt + 1, exc)
                    failures.append(short_reason(provider.name, exc))
                    if not is_transient(exc):
                        break  # same request, same error: go straight to the next provider
                    if attempt < self.retries:
                        self._sleep(self.backoff_s * 2**attempt)
                    continue
                completion.failures = failures
                return completion
        raise ProviderError("all providers failed: " + "; ".join(failures))


def is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(word in text for word in TRANSIENT)


def short_reason(name: str, exc: Exception) -> str:
    """'gemini:x failed: 504 DEADLINE_EXCEEDED. {...}' -> 'gemini:x: 504 DEADLINE_EXCEEDED'"""
    detail = str(exc).split(" failed: ", 1)[-1].split(".")[0]
    return f"{name}: {detail[:40]}"
