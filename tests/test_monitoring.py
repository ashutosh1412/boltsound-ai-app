from app.monitoring import summarize


def record(ms: int, route: str = "question", steps=()) -> dict:
    return {
        "ms": ms,
        "route": route,
        "cost_usd": 0.001,
        "steps": [{"name": n, "status": s} for n, s in steps],
    }


def test_summary_counts_latency_cost_and_handovers():
    records = [record(1000), record(2000), record(9000, steps=[("handover", "warn")]), record(500, "attack")]
    summary = summarize(records)
    assert summary.messages == 4
    assert summary.p50_ms == 1500
    assert summary.handover_rate == 0.25
    assert summary.routes == {"question": 3, "attack": 1}
