"""Estimated prices in US dollars per million tokens (input, output).

Check each provider's pricing page before relying on these; they change.
"""

PRICES_PER_M = {
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-flash-lite-latest": (0.10, 0.40),
    "gpt-5-mini": (0.25, 2.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_PER_M.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1e6
