"""Monitoring: one JSON line per answered message, so we can see speed, cost and failures over time.

Only the MASKED message is logged. Logs are read by many people; card numbers must never land there.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.types import Reply

REQUESTS_FILE = Path(__file__).resolve().parents[1] / "logs" / "requests.jsonl"


class RequestLog:
    """In real life these lines go to a log/trace system (e.g. OpenTelemetry, Langfuse, Datadog)."""

    def __init__(self, path: Path = REQUESTS_FILE) -> None:
        self.path = path

    def record(self, customer: str, reply: Reply) -> None:
        line = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "customer": customer,
            "message": reply.masked_message,
            "route": reply.route,
            "ms": round(reply.total_ms),
            "cost_usd": round(reply.cost_usd, 6),
            "steps": [{"name": s.name, "status": s.status, "ms": round(s.ms)} for s in reply.trace],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]


@dataclass(frozen=True)
class Summary:
    messages: int
    p50_ms: float
    p95_ms: float
    cost_usd: float
    handover_rate: float
    blocked: int
    errors: int
    routes: dict[str, int]


def summarize(records: list[dict]) -> Summary:
    def has(record: dict, name: str = "", status: str = "") -> bool:
        return any(
            (not name or s["name"] == name) and (not status or s["status"] == status) for s in record["steps"]
        )

    ms = [r["ms"] for r in records] or [0]
    routes: dict[str, int] = {}
    for r in records:
        routes[r["route"]] = routes.get(r["route"], 0) + 1
    return Summary(
        messages=len(records),
        p50_ms=float(np.percentile(ms, 50)),
        p95_ms=float(np.percentile(ms, 95)),
        cost_usd=sum(r["cost_usd"] for r in records),
        handover_rate=sum(has(r, "handover") for r in records) / max(len(records), 1),
        blocked=sum(has(r, status="blocked") for r in records),
        errors=sum(has(r, status="error") for r in records),
        routes=routes,
    )
