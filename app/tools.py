"""Tools: live data the model can't know, like the status of an order."""

import json
import re
from pathlib import Path

ORDERS_FILE = Path(__file__).resolve().parents[1] / "data" / "orders.json"
ORDER_ID_PATTERN = re.compile(r"\bbs-?(\d{4})\b", re.IGNORECASE)  # "BS-2291", "bs2291", "Bs-2291"


class OrderStore:
    def __init__(self, path: Path = ORDERS_FILE) -> None:
        self._orders: dict[str, dict] = json.loads(path.read_text())

    def find_order(self, order_id: str, customer: str) -> dict | None:
        """Return the order only if it belongs to this customer (never leak other people's orders)."""
        order = self._orders.get(order_id)
        return order if order and order["customer"] == customer else None


def find_order_ids(message: str) -> list[str]:
    return [f"BS-{digits}" for digits in ORDER_ID_PATTERN.findall(message)]
