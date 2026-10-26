"""Guardrails: checks on what goes into the model and what comes out of it."""

import re

PERSONAL_DATA_PATTERNS = {
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "phone": re.compile(r"(?:\+91[ -]?)?\b[6-9]\d{4}[ -]?\d{5}\b"),
}

ATTACK_PHRASES = (
    "ignore all previous instructions",
    "ignore your instructions",
    "ignore previous instructions",
    "system override",
    "you are now",
    "reveal your instructions",
    "print your hidden instructions",
)


def mask_personal_data(text: str) -> tuple[str, list[str]]:
    """Replace phone numbers, emails and card numbers before anything leaves our servers."""
    found = []
    for kind, pattern in PERSONAL_DATA_PATTERNS.items():  # cards first, so their digits aren't read as phones
        text, count = pattern.subn(f"[{kind.upper()}]", text)
        found += [kind] * count
    return text, found


def looks_like_prompt_attack(text: str) -> bool:
    """A cheap first filter. In production, add a guard model trained to spot attacks."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in ATTACK_PHRASES)
