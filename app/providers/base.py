"""The one interface every model provider implements."""

from typing import Protocol

from app.types import Completion


class ProviderError(Exception):
    """A provider call failed (rate limit, outage, bad request...)."""


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> Completion:
        """Return the model's reply to `prompt`, following the `system` instructions."""
        ...
