"""Returns: an action that moves money, so the rules live in code and the customer must confirm."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

RETURN_WINDOW_DAYS = 30
RETURNS_FILE = Path(__file__).resolve().parents[1] / "logs" / "returns.jsonl"
CONFIRM_WORDS = {"yes", "y", "confirm", "yes please", "yes confirm"}


@dataclass(frozen=True)
class ReturnRequest:
    order_id: str
    customer: str
    product: str
    amount_inr: int


def reason_cannot_return(order: dict) -> str | None:
    """None if the order can be returned, otherwise the reason. The model never decides this."""
    if order["status"] != "delivered":
        return "it hasn't been delivered yet"
    days = order["delivered_days_ago"]
    if days > RETURN_WINDOW_DAYS:
        return f"it was delivered {days} days ago (returns are within {RETURN_WINDOW_DAYS} days)"
    return None


def is_confirmation(message: str) -> bool:
    return message.strip().lower().rstrip(".!") in CONFIRM_WORDS


class ReturnDesk:
    """Creates return requests. In real life: the order system's API."""

    def __init__(self, path: Path = RETURNS_FILE) -> None:
        self.path = path

    def existing(self, order_id: str) -> str | None:
        """The return already created for this order, if any (never refund the same order twice)."""
        if not self.path.exists():
            return None
        for line in self.path.read_text().splitlines():
            record = json.loads(line)
            if record["order_id"] == order_id:
                return record["id"]
        return None

    def create(self, request: ReturnRequest) -> str:
        if existing := self.existing(request.order_id):
            return existing
        return_id = f"RT-{uuid4().hex[:6].upper()}"
        record = {"id": return_id, **asdict(request), "created_at": datetime.now(timezone.utc).isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return return_id
