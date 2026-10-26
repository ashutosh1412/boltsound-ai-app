"""OpenAI, through the official openai SDK (Responses API)."""

from openai import OpenAI

from app.pricing import estimate_cost
from app.providers.base import ProviderError
from app.types import Completion


class OpenAIProvider:
    def __init__(self, api_key: str, model: str, timeout_s: float = 30) -> None:
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self.name = f"openai:{model}"
        self._model = model
        self._client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)

    def complete(self, system: str, prompt: str) -> Completion:
        try:
            response = self._client.responses.create(model=self._model, instructions=system, input=prompt)
        except Exception as exc:
            raise ProviderError(f"{self.name} failed: {exc}") from exc
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        return Completion(
            text=response.output_text or "",
            model=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(self._model, input_tokens, output_tokens),
        )
