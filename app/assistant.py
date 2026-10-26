"""BoltSound's support assistant: every customer message goes through `Assistant.answer`."""

import time
from typing import Protocol

from app.config import Settings
from app.providers.base import LLMProvider
from app.retrieval import Chunk
from app.types import Completion, Reply, TraceStep

SYSTEM_PROMPT = (
    "You are BoltSound's friendly customer support assistant. "
    "BoltSound sells wireless headphones in India. Answer in at most 3 short sentences. "
    "Use ONLY the help-centre extracts provided. If they don't contain the answer, say you'll "
    "connect the customer to a human agent. End with the source file in brackets, e.g. [returns.md]."
)


class Searcher(Protocol):
    def search(self, question: str) -> list[Chunk]: ...


class Assistant:
    def __init__(self, provider: LLMProvider, settings: Settings, retriever: Searcher) -> None:
        self.provider = provider
        self.settings = settings
        self.retriever = retriever

    def answer(self, message: str) -> Reply:
        reply = Reply(text="")
        chunks = self._retrieve(message, reply)
        prompt = build_prompt(message, chunks)
        completion = self._call_model(SYSTEM_PROMPT, prompt, reply)
        reply.text = completion.text.strip()
        return reply

    def _retrieve(self, message: str, reply: Reply) -> list[Chunk]:
        start = time.perf_counter()
        chunks = self.retriever.search(message)
        sources = ", ".join(c.source for c in chunks) or "nothing found"
        reply.trace.append(TraceStep("retrieval", sources, ms=(time.perf_counter() - start) * 1000))
        return chunks

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


def build_prompt(message: str, chunks: list[Chunk]) -> str:
    extracts = "\n\n".join(f"[{chunk.source}]\n{chunk.text}" for chunk in chunks)
    return f"Help-centre extracts:\n{extracts}\n\nCustomer: {message}"
