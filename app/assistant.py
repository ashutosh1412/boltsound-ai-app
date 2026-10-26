"""BoltSound's support assistant: every customer message goes through `Assistant.answer`."""

import time
from typing import Protocol

import numpy as np

from app.cache import AnswerCache
from app.config import Settings
from app.guardrails import (
    check_output,
    looks_like_prompt_attack,
    mask_personal_data,
    wants_handover,
)
from app.handoff import HandoffQueue
from app.providers.base import LLMProvider, ProviderError
from app.retrieval import Chunk
from app.router import ROUTER_PROMPT, Route, parse_route
from app.tools import OrderStore, find_order_ids
from app.types import Completion, Reply, TraceStep

BLOCKED_REPLY = "Sorry, I can't help with that. I can help with orders, returns, warranty and our products."
SMALL_TALK_REPLY = (
    "Hi! I'm Bolt, BoltSound's assistant. I can help with orders, returns, warranty and our products."
)

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
    def embed(self, text: str) -> np.ndarray: ...

    def search(self, query: np.ndarray) -> list[Chunk]: ...


class Assistant:
    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        retriever: Searcher,
        orders: OrderStore,
        router: LLMProvider,
        handoffs: HandoffQueue,
        cache: AnswerCache,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.handoffs = handoffs
        self.router = router
        self.settings = settings
        self.retriever = retriever
        self.orders = orders

    def answer(self, message: str, customer: str) -> Reply:
        reply = Reply(text="")
        message, route = self._guard_and_route(message, reply)
        if route is Route.ATTACK:
            reply.text = BLOCKED_REPLY
            return reply
        if route is Route.SMALL_TALK:
            reply.text = SMALL_TALK_REPLY
            return reply
        if route is Route.ORDER_STATUS and (status := self._order_status(message, customer, reply)):
            reply.text = status
            return reply
        reply.text = self._answer_question(message, customer, reply)
        return reply

    def _answer_question(self, message: str, customer: str, reply: Reply) -> str:
        """The full path: cache → retrieval + order data → model → output guard → (handover)."""
        order_facts = self._look_up_orders(message, customer, reply)
        vector = self.retriever.embed(message)
        # Only general answers are shared. An answer built from someone's order is theirs alone.
        shareable = not order_facts
        if shareable and (hit := self.cache.get(message, vector)):
            reply.cost_usd += hit.cost_usd
            reply.trace.append(TraceStep("cache", f"hit: {hit.how}, no big-model call"))
            return hit.answer
        chunks = self._retrieve(vector, reply)
        prompt = build_prompt(message, chunks, order_facts)
        allowed = {chunk.source for chunk in chunks} | ({"orders"} if order_facts else set())
        start = time.perf_counter()
        try:
            answer = self._answer_checked(prompt, allowed, reply)
            reason = "not covered by help centre"
        except ProviderError as exc:
            ms = (time.perf_counter() - start) * 1000
            reply.trace.append(TraceStep("model", str(exc), ms=ms, status="error"))
            answer, reason = "", "models unavailable"
        if wants_handover(answer) or not answer:
            ticket = self.handoffs.create(customer, message, reason=reason)
            reply.trace.append(TraceStep("handover", f"ticket {ticket.id} created", status="warn"))
            return HANDOVER_REPLY.format(ticket=ticket.id)
        if shareable:
            self.cache.put(message, vector, answer)
        return answer

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

    def _guard_and_route(self, message: str, reply: Reply) -> tuple[str, Route]:
        start = time.perf_counter()
        masked, found = mask_personal_data(message)
        if looks_like_prompt_attack(masked):
            route = Route.ATTACK
        else:
            decision = self.router.complete(ROUTER_PROMPT, masked)
            reply.cost_usd += decision.cost_usd
            route = parse_route(decision.text)
        ms = (time.perf_counter() - start) * 1000
        if route is Route.ATTACK:
            reply.trace.append(TraceStep("input guard", "prompt attack blocked", ms=ms, status="blocked"))
        else:
            detail = f"masked: {', '.join(found)} · " if found else ""
            reply.trace.append(
                TraceStep(
                    "input guard", detail + f"route: {route.value}", ms=ms, status="warn" if found else "ok"
                )
            )
        return masked, route

    def _order_status(self, message: str, customer: str, reply: Reply) -> str | None:
        """Simple 'where is my order?' needs no big model: answer from the order data."""
        lines = []
        for order_id in find_order_ids(message):
            order = self.orders.find_order(order_id, customer)
            if order is None:
                return None  # unknown or someone else's order: let the full path handle it
            lines.append(f"Your order {order_id} ({order['product']}) is {order['status']}.")
        if not lines:
            return None
        reply.trace.append(TraceStep("tool:get_order", "answered from order data, no model call"))
        return " ".join(lines) + " Track it at boltsound.in/track. [orders]"

    def _retrieve(self, vector: np.ndarray, reply: Reply) -> list[Chunk]:
        start = time.perf_counter()
        chunks = self.retriever.search(vector)
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
        reply.cost_usd += completion.cost_usd
        detail = f"{completion.model} · {completion.input_tokens}→{completion.output_tokens} tokens"
        if completion.failures:
            detail += " · after failures: " + ", ".join(completion.failures)
        ms = (time.perf_counter() - start) * 1000
        reply.trace.append(TraceStep("model", detail, ms=ms, status="warn" if completion.failures else "ok"))
        return completion


def build_prompt(message: str, chunks: list[Chunk], order_facts: list[str] | None = None) -> str:
    extracts = "\n\n".join(f"[{chunk.source}]\n{chunk.text}" for chunk in chunks)
    orders = "\n".join(order_facts or []) or "none"
    return f"Help-centre extracts:\n{extracts}\n\nOrder data [orders]:\n{orders}\n\nCustomer: {message}"
