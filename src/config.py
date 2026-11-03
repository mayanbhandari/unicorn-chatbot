# Central configuration for the whole bot: paths, model choices and tunable limits in one place. we need it as every other module reads its settings from here, so changing a model or a threshold never means hunting through the code.

from __future__ import annotations  # lets us use modern type hints (like `str | None`) safely

import os                                    # to read environment variables
from dataclasses import dataclass, field     # dataclass = a tidy class for holding settings; field = custom defaults
from pathlib import Path                     # cleaner, cross-platform file paths than plain strings

from dotenv import load_dotenv               # loads the key=value pairs from our .env file

# Project root = two folders up from this file (src/config.py -> src -> project root).
ROOT = Path(__file__).resolve().parent.parent
# Read .env once at import time so the API keys and overrides are available below.
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    # Return the env variable if it has a real value, otherwise the default.
    # WHY `or` and not getenv(name, default): a blank line like `CHAT_MODEL=` in .env is an EMPTY string,
    # and getenv would hand that back as if it were a real model name. `or` treats blank as "not set".
    return os.getenv(name) or default   # empty value in .env counts as unset


# Sensible defaults for each supported provider: (chat model, embedding model).
PROVIDER_DEFAULTS = {
    # provider: (chat model, embedding model)  -> override with CHAT_MODEL / EMBEDDING_MODEL in .env
    "gemini": ("gemini-3.5-flash", "models/gemini-embedding-001"),   # free tier, our default
    "ollama": ("llama3.1:8b", "nomic-embed-text"),                      # free, runs locally on the laptop
    "openai": ("gpt-4o-mini", "text-embedding-3-small"),                # paid option
}


def _provider() -> str:
    # Which provider the user picked; defaults to Gemini, and we lowercase/trim so "Gemini " still works.
    provider = _env("LLM_PROVIDER", "gemini").strip().lower()
    # Fail fast with a clear message if someone typos the provider, instead of crashing deep inside LangChain later.
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDER_DEFAULTS)}, got {provider!r}")
    return provider


@dataclass(frozen=True)   # frozen = read-only once created, so no part of the app can change settings by accident
class Settings:
    # Paths
    data_path: Path = ROOT / "data" / "tracxn.csv"    # the static dataset we answer from
    chroma_dir: Path = ROOT / "data" / "chroma"       # where the saved embedding index lives
    log_dir: Path = ROOT / "logs"                     # where the JSON log file is written

    # Model provider + models (override in .env)
    # default_factory runs when a Settings object is created, so it reads the CURRENT .env value each time.
    llm_provider: str = field(default_factory=_provider)
    # Chat model: use CHAT_MODEL from .env if set, otherwise the provider's default (index 0 of the tuple).
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", PROVIDER_DEFAULTS[_provider()][0]))
    # Embedding model: same idea, index 1 of the tuple is the embedding default.
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", PROVIDER_DEFAULTS[_provider()][1]))

    # Conversation / retrieval behaviour
    max_input_chars: int = 500      # longer user messages get cut off (basic input protection)
    history_turns: int = 6          # past user+bot turns sent to the router
    top_k: int = 5                  # semantic recommendations returned
    max_list_items: int = 30        # cap for filter/list answers
    max_clarifications: int = 2     # consecutive clarifying questions before we just answer
    fx_inr_per_usd: float = 85.0    # only used to sort/compare valuations


def get_settings() -> Settings:
    # One small helper so the rest of the code just calls get_settings() and never builds Settings by hand.
    return Settings()