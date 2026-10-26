"""Tools: live data the model can't know, like the status of an order."""

import json
import re
from pathlib import Path

ORDERS_FILE = Path(__file__).resolve().parents[1] / "data" / "orders.json"
ORDER_ID_PATTERN = re.compile(r"\bBS-\d{4}\b")


class OrderStore:
    def __init__(self, path: Path = ORDERS_FILE) -> None:
        self._orders: dict[str, dict] = json.loads(path.read_text())

    def find_order(self, order_id: str) -> dict | None:
        return self._orders.get(order_id)


def find_order_ids(message: str) -> list[str]:
    return ORDER_ID_PATTERN.findall(message)
