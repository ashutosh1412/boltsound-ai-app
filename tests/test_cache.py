import numpy as np

from app.cache import AnswerCache
from app.providers.fake import FakeProvider


def unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=float)
    return v / np.linalg.norm(v)


def cache_with(judge_says: str = "yes") -> AnswerCache:
    cache = AnswerCache(judge=FakeProvider(judge_says), threshold=0.75)
    cache.put("Is the X1 waterproof?", unit(1, 0), "No, keep it dry. [x1-manual.md]")
    return cache


def test_exact_match_ignores_case_and_punctuation():
    assert cache_with().get("is the x1 waterproof", unit(0, 1)).how == "exact"


def test_confirmed_paraphrase_hits():
    hit = cache_with("yes").get("Can I swim with my X1?", unit(1, 0.3))
    assert hit is not None and hit.how.endswith("confirmed")


def test_different_product_never_hits_even_when_similar():
    assert cache_with("yes").get("Is the X2 waterproof?", unit(1, 0.1)) is None


def test_same_product_different_question_is_rejected_by_the_judge():
    assert cache_with("no").get("Does the X1 have a warranty?", unit(1, 0.3)) is None
