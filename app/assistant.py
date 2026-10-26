"""BoltSound's support assistant: every customer message goes through `Assistant.answer`."""

import time
from typing import Protocol

from app.config import Settings
from app.guardrails import (
    check_output,
    guard_model_says_attack,
    looks_like_prompt_attack,
    mask_personal_data,
    wants_handover,
)
from app.handoff import HandoffQueue
from app.providers.base import LLMProvider
from app.retrieval import Chunk
from app.tools import OrderStore, find_order_ids
from app.types import Completion, Reply, TraceStep

BLOCKED_REPLY = "Sorry, I can't help with that. I can help with orders, returns, warranty and our products."

SYSTEM_PROMPT = (
    "You are BoltSound's friendly customer support assistant. "
    "BoltSound sells wireless headphones in India. Answer in at most 3 short sentences. "
    "Use ONLY the help-centre extracts provided. If they don't contain the answer, say you'll "
    "connect the customer to a human agent. End with the source file in brackets, e.g. [returns.md]."
)

RETRY_HINT = "\n\nYour last answer was rejected ({problem}). Answer again and end with one of: {sources}."
HANDOVER_REPLY = (
    "I've passed this to our support team (ticket {ticket}). A person will reply within 24 hours."
)


class Searcher(Protocol):
    def search(self, question: str) -> list[Chunk]: ...


class Assistant:
    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        retriever: Searcher,
        orders: OrderStore,
        guard: LLMProvider,
        handoffs: HandoffQueue,
    ) -> None:
        self.provider = provider
        self.handoffs = handoffs
        self.guard = guard
        self.settings = settings
        self.retriever = retriever
        self.orders = orders

    def answer(self, message: str, customer: str) -> Reply:
        reply = Reply(text="")
        message, allowed = self._guard_input(message, reply)
        if not allowed:
            reply.text = BLOCKED_REPLY
            return reply
        chunks = self._retrieve(message, reply)
        order_facts = self._look_up_orders(message, customer, reply)
        prompt = build_prompt(message, chunks, order_facts)
        allowed = {chunk.source for chunk in chunks} | ({"orders"} if order_facts else set())
        reply.text = self._answer_checked(prompt, allowed, reply)
        if wants_handover(reply.text) or not reply.text:
            ticket = self.handoffs.create(customer, message, reason="not covered by help centre")
            reply.trace.append(TraceStep("handover", f"ticket {ticket.id} created", status="warn"))
            reply.text = HANDOVER_REPLY.format(ticket=ticket.id)
        return reply

    def _answer_checked(self, prompt: str, allowed: set[str], reply: Reply) -> str:
        """Call the model, check its answer, retry once with feedback, else hand over ('' = give up)."""
        for attempt in range(2):
            text = self._call_model(SYSTEM_PROMPT, prompt, reply).text.strip()
            if wants_handover(text):
                return text
            problem = check_output(text, allowed)
            if problem is None:
                reply.trace.append(TraceStep("output guard", "source checked"))
                return text
            status = "warn" if attempt == 0 else "blocked"
            reply.trace.append(TraceStep("output guard", problem, status=status))
            prompt += RETRY_HINT.format(problem=problem, sources=", ".join(sorted(allowed)) or "none")
        return ""

    def _guard_input(self, message: str, reply: Reply) -> tuple[str, bool]:
        start = time.perf_counter()
        masked, found = mask_personal_data(message)
        if looks_like_prompt_attack(masked) or guard_model_says_attack(self.guard, masked):
            ms = (time.perf_counter() - start) * 1000
            reply.trace.append(TraceStep("input guard", "prompt attack blocked", ms=ms, status="blocked"))
            return masked, False
        detail = f"masked: {', '.join(found)}" if found else "clean"
        ms = (time.perf_counter() - start) * 1000
        reply.trace.append(TraceStep("input guard", detail, ms=ms, status="warn" if found else "ok"))
        return masked, True

    def _retrieve(self, message: str, reply: Reply) -> list[Chunk]:
        start = time.perf_counter()
        chunks = self.retriever.search(message)
        sources = ", ".join(sorted({chunk.source for chunk in chunks})) or "nothing found"
        reply.trace.append(TraceStep("retrieval", sources, ms=(time.perf_counter() - start) * 1000))
        return chunks

    def _look_up_orders(self, message: str, customer: str, reply: Reply) -> list[str]:
        facts = []
        for order_id in find_order_ids(message):
            order = self.orders.find_order(order_id, customer)
            facts.append(f"{order_id}: {order}" if order else f"{order_id}: not found for {customer}")
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
