"""BoltSound's support assistant: every customer message goes through `Assistant.answer`."""

import time

from app.config import Settings
from app.providers.base import LLMProvider
from app.types import Completion, Reply, TraceStep

SYSTEM_PROMPT = (
    "You are BoltSound's friendly customer support assistant. "
    "BoltSound sells wireless headphones in India. Answer in at most 3 short sentences."
)


class Assistant:
    def __init__(self, provider: LLMProvider, settings: Settings) -> None:
        self.provider = provider
        self.settings = settings

    def answer(self, message: str) -> Reply:
        reply = Reply(text="")
        completion = self._call_model(SYSTEM_PROMPT, message, reply)
        reply.text = completion.text.strip()
        return reply

    def _call_model(self, system: str, prompt: str, reply: Reply) -> Completion:
        start = time.perf_counter()
        completion = self.provider.complete(system, prompt)
        reply.cost_usd += self._cost(completion)
        reply.trace.append(
            TraceStep(
                "model",
                f"{completion.model} · {completion.input_tokens}→{completion.output_tokens} tokens",
                ms=(time.perf_counter() - start) * 1000,
            )
        )
        return completion

    def _cost(self, completion: Completion) -> float:
        settings = self.settings
        return (
            completion.input_tokens * settings.price_in_per_m
            + completion.output_tokens * settings.price_out_per_m
        ) / 1e6
