"""Answer cache: the same (or the same-meaning) general question shouldn't cost a big-model call twice.

Lesson from the build log: embedding similarity means "same topic", not "same question".
"Is the X2 waterproof?" scored 0.84 against "Is the X1 waterproof?" - higher than a real
paraphrase (0.77). So a similar hit must also mention the same products and be confirmed
by a small model before we reuse it.
"""

import re
from dataclasses import dataclass

import numpy as np

from app.providers.base import LLMProvider

PRODUCTS = re.compile(r"\b(x1|x2|buds pro)\b", re.IGNORECASE)
SAME_QUESTION_PROMPT = (
    "Do these two customer questions ask for exactly the same information about the same product? "
    "Reply with one word: yes or no."
)


def normalise(question: str) -> str:
    return re.sub(r"[^\w\s]", "", question.lower()).strip()


def product_names(question: str) -> set[str]:
    return {name.lower() for name in PRODUCTS.findall(question)}


@dataclass(frozen=True)
class CacheHit:
    answer: str
    how: str
    cost_usd: float = 0.0  # the judge call, if one was needed


@dataclass(frozen=True)
class CachedAnswer:
    question: str
    vector: np.ndarray
    answer: str


class AnswerCache:
    def __init__(self, judge: LLMProvider, threshold: float = 0.75) -> None:
        self.judge = judge
        self.threshold = threshold
        self._exact: dict[str, str] = {}
        self._entries: list[CachedAnswer] = []

    def get(self, question: str, vector: np.ndarray) -> CacheHit | None:
        if (answer := self._exact.get(normalise(question))) is not None:
            return CacheHit(answer, "exact")
        similarities = [(float(entry.vector @ vector), entry) for entry in self._entries]
        for score, entry in sorted(similarities, key=lambda pair: -pair[0])[:1]:
            if score < self.threshold or product_names(entry.question) != product_names(question):
                return None
            verdict = self.judge.complete(SAME_QUESTION_PROMPT, f"1: {entry.question}\n2: {question}")
            if verdict.text.strip().lower().startswith("yes"):
                return CacheHit(entry.answer, f"similar ({score:.2f}), confirmed", verdict.cost_usd)
        return None

    def put(self, question: str, vector: np.ndarray, answer: str) -> None:
        self._exact[normalise(question)] = answer
        self._entries.append(CachedAnswer(question, vector, answer))
