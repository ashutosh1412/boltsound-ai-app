from app.router import Route, parse_route


def test_reads_the_label():
    assert parse_route(" Order_Status\n") is Route.ORDER_STATUS


def test_unclear_label_takes_the_safe_full_path():
    assert parse_route("I think this is about headphones") is Route.QUESTION
