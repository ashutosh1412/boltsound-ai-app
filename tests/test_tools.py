from app.tools import OrderStore, find_order_ids


def test_find_order_ids_accepts_common_spellings():
    assert find_order_ids("Where is BS-2291, bs4102 and Bs-3310?") == ["BS-2291", "BS-4102", "BS-3310"]


def test_customers_only_see_their_own_orders():
    store = OrderStore()
    assert store.find_order("BS-2291", "Riya")["product"] == "BoltSound X2"
    assert store.find_order("BS-3310", "Riya") is None  # Arjun's order
