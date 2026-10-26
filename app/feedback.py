"""Feedback: what customers tell us (thumbs) and what their behaviour tells us (implicit signals).

Traps to remember: very few people press a button, and mostly when they are unhappy; a "thanks"
can follow a wrong answer. So read thumbs and behaviour together, and check samples by hand.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

FRUSTRATION = ("that's wrong", "thats wrong", "not helpful", "useless", "real person", "human", "agent")


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def implicit_signal(previous: str | None, current: str) -> str | None:
    """'frustrated' if the customer sounds unhappy, 'rephrased' if they ask almost the same thing again."""
    lowered = current.lower()
    if any(phrase in lowered for phrase in FRUSTRATION):
        return "frustrated"
    if previous:
        before, now = words(previous), words(current)
        overlap = len(before & now) / max(len(before | now), 1)  # share of words in common
        if overlap >= 0.5 and current.strip().lower() != previous.strip().lower():
            return "rephrased"
    return None


class FeedbackLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, customer: str, message: str, signal: str, kind: str) -> None:
        line = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "customer": customer,
            "message": message,  # already masked by the caller
            "signal": signal,  # up | down | frustrated | rephrased
            "kind": kind,  # explicit | implicit
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]
