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
from app.returns import ReturnDesk, ReturnRequest, is_confirmation, reason_cannot_return
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
RETURN_OFFER = (
    "I can start a return for {order_id} ({product}, ₹{amount:,}). After we receive it, the refund goes to "
    "your original payment method within 5–7 working days. Reply YES to confirm. [returns.md]"
)
RETURN_DONE = (
    "Done! Return {return_id} is created for {order_id}. We'll email you a pickup slot. [returns.md]"
)
RETURN_REFUSED = (
    "Sorry, {order_id} can't be returned: {reason}. Warranty claims are still possible. [returns.md]"
)
ALREADY_RETURNED = "{order_id} already has a return in progress ({return_id}). [orders]"
WHICH_ORDER = "Which order would you like to return? Orders on your account: {orders}. [orders]"
NOT_YOUR_ORDER = "I can't find {order_id} on your account. Please check the order number. [orders]"
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
        returns: ReturnDesk,
    ) -> None:
        self.provider = provider
        self.returns = returns
        # Returns waiting for the customer's YES. In real life: a session store such as Redis.
        self._pending: dict[str, ReturnRequest] = {}
        self.cache = cache
        self.handoffs = handoffs
        self.router = router
        self.settings = settings
        self.retriever = retriever
        self.orders = orders

    def answer(self, message: str, customer: str) -> Reply:
        reply = Reply(text="")
        pending = self._pending.pop(customer, None)  # an offer is only valid for the very next message
        if pending and is_confirmation(message):
            return_id = self.returns.create(pending)
            reply.trace.append(TraceStep("action:create_return", f"{return_id} for {pending.order_id}"))
            reply.text = RETURN_DONE.format(return_id=return_id, order_id=pending.order_id)
            return reply
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
        if route is Route.RETURN_REQUEST and (offer := self._offer_return(message, customer, reply)):
            reply.text = offer
            return reply
        reply.text = self._answer_question(message, customer, reply)
        return reply

    def _offer_return(self, message: str, customer: str, reply: Reply) -> str | None:
        """Check the rules in code, then ask the customer to confirm. Nothing happens without YES."""
        order_ids = find_order_ids(message)
        if not order_ids:
            return WHICH_ORDER.format(orders=", ".join(self.orders.order_ids_for(customer)) or "none")
        if len(order_ids) > 1:
            return None  # several orders in one message: let the normal path explain
        order_id = order_ids[0]
        if (order := self.orders.find_order(order_id, customer)) is None:
            reply.trace.append(
                TraceStep("tool:get_order", f"{order_id}: not on {customer}'s account", status="warn")
            )
            return NOT_YOUR_ORDER.format(order_id=order_id)
        if return_id := self.returns.existing(order_id):
            return ALREADY_RETURNED.format(order_id=order_id, return_id=return_id)
        if reason := reason_cannot_return(order):
            reply.trace.append(TraceStep("rule:return_window", f"{order_id}: {reason}", status="blocked"))
            return RETURN_REFUSED.format(order_id=order_id, reason=reason)
        self._pending[customer] = ReturnRequest(order_id, customer, order["product"], order["price"])
        reply.trace.append(TraceStep("rule:return_window", f"{order_id}: eligible, waiting for YES"))
        return RETURN_OFFER.format(order_id=order_id, product=order["product"], amount=order["price"])

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
