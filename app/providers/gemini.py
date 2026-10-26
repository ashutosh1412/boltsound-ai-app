"""Google Gemini, through the official google-genai SDK."""

from google import genai
from google.genai import types

from app.providers.base import ProviderError
from app.types import Completion


class GeminiProvider:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is not set")
        self.name = f"gemini:{model}"
        self._model = model
        self._client = genai.Client(api_key=api_key)

    def complete(self, system: str, prompt: str) -> Completion:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.2),
            )
        except Exception as exc:  # the SDK raises several error types
            raise ProviderError(f"{self.name} failed: {exc}") from exc
        usage = response.usage_metadata
        return Completion(
            text=response.text or "",
            model=self.name,
            input_tokens=usage.prompt_token_count or 0,
            output_tokens=usage.candidates_token_count or 0,
        )
