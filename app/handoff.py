"""Human handover: when the bot can't answer, a real person must actually get the case."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HANDOFF_FILE = Path(__file__).resolve().parents[1] / "logs" / "handoffs.jsonl"


@dataclass(frozen=True)
class Ticket:
    id: str
    customer: str
    message: str
    reason: str
    created_at: str


class HandoffQueue:
    """A file-backed queue. In real life this would create a ticket in Zendesk, Freshdesk, etc."""

    def __init__(self, path: Path = HANDOFF_FILE) -> None:
        self.path = path

    def create(self, customer: str, message: str, reason: str) -> Ticket:
        ticket = Ticket(
            id=f"HT-{uuid4().hex[:6].upper()}",
            customer=customer,
            message=message,
            reason=reason,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(ticket)) + "\n")
        return ticket
