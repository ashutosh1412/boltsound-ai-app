"""Settings, read from environment variables (a local .env file works too)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    openai_api_key: str
    main_model: str
    # Prices in US dollars per million tokens, used only for cost estimates in the UI.
    price_in_per_m: float
    price_out_per_m: float


def load_settings() -> Settings:
    """Build Settings from the environment. Keys never live in the code."""
    load_dotenv()
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        main_model=os.getenv("MAIN_MODEL", "gemini-flash-latest"),
        price_in_per_m=float(os.getenv("PRICE_IN_PER_M", "0.30")),
        price_out_per_m=float(os.getenv("PRICE_OUT_PER_M", "2.50")),
    )
