"""A scripted provider for tests: no network, no cost."""

from collections.abc import Callable

from app.providers.base import ProviderError
from app.types import Completion


class FakeProvider:
    def __init__(self, reply: str | Callable[[str, str], str] = "OK", name: str = "fake") -> None:
        self.name = name
        self._reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> Completion:
        self.calls.append((system, prompt))
        text = self._reply(system, prompt) if callable(self._reply) else self._reply
        return Completion(
            text=text, model=self.name, input_tokens=len(prompt) // 4, output_tokens=len(text) // 4
        )


class DownProvider:
    """Always fails, like a provider during an outage."""

    def __init__(self, name: str = "down") -> None:
        self.name = name
        self.calls = 0

    def complete(self, system: str, prompt: str):
        self.calls += 1
        raise ProviderError("503 service unavailable")
