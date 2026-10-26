from app.tools import OrderStore, find_order_ids


def test_find_order_ids():
    assert find_order_ids("Where is BS-2291 and BS-4102?") == ["BS-2291", "BS-4102"]


def test_unknown_order_returns_none():
    assert OrderStore().find_order("BS-0000") is None
