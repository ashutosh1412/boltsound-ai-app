"""Google Gemini, through the official google-genai SDK."""

from google import genai
from google.genai import types

from app.pricing import estimate_cost
from app.providers.base import ProviderError
from app.types import Completion


class GeminiProvider:
    def __init__(self, api_key: str, model: str, timeout_s: float = 30, thinking: str = "low") -> None:
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is not set")
        self.name = f"gemini:{model}"
        self._model = model
        self._thinking = types.ThinkingConfig(thinking_level=thinking)
        # No hidden SDK retries: the gateway decides when to retry or switch provider.
        options = types.HttpOptions(
            timeout=int(timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1)
        )
        self._client = genai.Client(api_key=api_key, http_options=options)

    def complete(self, system: str, prompt: str) -> Completion:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system, temperature=0.2, thinking_config=self._thinking
                ),
            )
        except Exception as exc:  # the SDK raises several error types
            raise ProviderError(f"{self.name} failed: {exc}") from exc
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        # Thinking tokens are billed as output too, so count them.
        output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return Completion(
            text=response.text or "",
            model=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(self._model, input_tokens, output_tokens),
        )
