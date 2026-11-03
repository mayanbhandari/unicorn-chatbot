# Writes one JSON line per event to logs/chat.jsonl (time, session, question, answer, latency...). We need it so every conversation can be debugged or analysed afterwards, and the assignment asks for structured logs.

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# one named logger for the whole bot, so we never mix with other libraries' logs
LOGGER_NAME = "unicorn_bot"


class JsonFormatter(logging.Formatter):
    # Turns each log record into one line of JSON instead of plain text, so tools can read it.
    def format(self, record: logging.LogRecord) -> str:
        # the fields every line has
        payload: dict[str, Any] = {
            # UTC timestamp with milliseconds, so lines sort and compare easily
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
        }
        # merge in whatever extra details this event carries (session_id, query, answer...)
        payload.update(getattr(record, "fields", {}))
        # errors include the full traceback as text
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # ensure_ascii=False keeps characters like ₹ readable; default=str stops odd types from crashing the logger
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """File-only handler so the CLI stays clean; call once at start-up."""
    # create the logs folder if it's missing
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    # don't pass our logs up to the root logger, otherwise they would print on screen too
    logger.propagate = False
    # only add the file handler once, even if this function is called again
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        handler = logging.FileHandler(log_dir / "chat.jsonl", encoding="utf-8")
        # use our JSON format for every line
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


def log_event(event: str, level: int = logging.INFO, exc_info: bool = False, **fields: Any) -> None:
    # The one-line helper the rest of the code uses: log_event("chat_turn", session_id=..., ...)
    logging.getLogger(LOGGER_NAME).log(level, event, extra={"event": event, "fields": fields}, exc_info=exc_info)