"""BoltSound's support assistant: every customer message goes through `Assistant.answer`."""

import time
from typing import Protocol

from app.config import Settings
from app.providers.base import LLMProvider
from app.retrieval import Chunk
from app.tools import OrderStore, find_order_ids
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
    def __init__(
        self, provider: LLMProvider, settings: Settings, retriever: Searcher, orders: OrderStore
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.retriever = retriever
        self.orders = orders

    def answer(self, message: str) -> Reply:
        reply = Reply(text="")
        chunks = self._retrieve(message, reply)
        order_facts = self._look_up_orders(message, reply)
        prompt = build_prompt(message, chunks, order_facts)
        completion = self._call_model(SYSTEM_PROMPT, prompt, reply)
        reply.text = completion.text.strip()
        return reply

    def _retrieve(self, message: str, reply: Reply) -> list[Chunk]:
        start = time.perf_counter()
        chunks = self.retriever.search(message)
        sources = ", ".join(sorted({chunk.source for chunk in chunks})) or "nothing found"
        reply.trace.append(TraceStep("retrieval", sources, ms=(time.perf_counter() - start) * 1000))
        return chunks

    def _look_up_orders(self, message: str, reply: Reply) -> list[str]:
        facts = []
        for order_id in find_order_ids(message):
            order = self.orders.find_order(order_id)
            facts.append(f"{order_id}: {order}" if order else f"{order_id}: no such order")
        if facts:
            reply.trace.append(TraceStep("tool:get_order", "; ".join(facts)))
        return facts

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


def build_prompt(message: str, chunks: list[Chunk], order_facts: list[str] | None = None) -> str:
    extracts = "\n\n".join(f"[{chunk.source}]\n{chunk.text}" for chunk in chunks)
    orders = "\n".join(order_facts or []) or "none"
    return f"Help-centre extracts:\n{extracts}\n\nOrder data [orders]:\n{orders}\n\nCustomer: {message}"
