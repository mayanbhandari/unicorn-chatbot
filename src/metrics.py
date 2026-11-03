# Keeps count of what the bot is doing (queries, clarifications, fallbacks, errors, speed) and sends traces to Langfuse when it's switched on. We need it so we can see how the bot behaves in practice, and so tracing can never break the chat if Langfuse is off or down.

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator

# a normal logger just for warnings from this file (the chat logs go through logging_setup.py)
logger = logging.getLogger("unicorn_bot.metrics")


class Metrics:
    # Simple in-memory counters. They reset when the program stops; /stats in the CLI prints them.
    def __init__(self) -> None:
        # the lock makes it safe if two threads ever update the counters at once
        self._lock = threading.Lock()
        # every counter starts at zero
        self.counters: dict[str, int] = {
            "queries": 0, "clarifications": 0, "blocked_inputs": 0, "out_of_scope": 0,
            "not_found": 0, "no_results": 0, "errors": 0,
        }
        # how many times each intent (company_info, filter_list...) was used
        self.intents: dict[str, int] = {}
        # every turn's response time, kept so we can work out average and p95
        self._latencies_ms: list[float] = []

    def incr(self, name: str, n: int = 1) -> None:
        # add to a counter (creates it if it doesn't exist yet)
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + n

    def record_turn(self, intent: str, latency_ms: float) -> None:
        # called once per finished turn: tally the intent and store the latency
        with self._lock:
            self.intents[intent] = self.intents.get(intent, 0) + 1
            self._latencies_ms.append(latency_ms)

    def snapshot(self) -> dict[str, Any]:
        # a ready-to-print summary of everything so far; this is what /stats shows
        with self._lock:
            lat = sorted(self._latencies_ms)
            # p95 = the response time that 95% of turns beat; shows the slow cases that the average hides
            p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0.0
            c = self.counters
            # Keep this aligned with BotReply.fallback: blocked/sanitized inputs also
            # return a safe fallback response, even though they are not model errors.
            fallbacks = (c["blocked_inputs"] + c["out_of_scope"] + c["not_found"]
                         + c["no_results"] + c["errors"])
            # all the counters, plus the numbers we work out here
            return {
                **c,
                "fallbacks": fallbacks,
                "clarification_rate": round(c["clarifications"] / c["queries"], 3) if c["queries"] else 0.0,
                "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
                "p95_latency_ms": round(p95, 1),
                "intents": dict(self.intents),
            }


# A do-nothing stand-in for a Langfuse span. When tracing is off we hand this out instead,
# so the rest of the code can call span.update(...) without checking whether Langfuse exists.
class _NullSpan:
    def update(self, **_: Any) -> None:
        pass


class Observability:
    """Langfuse wrapper. `enabled` is False when keys are absent or init fails."""

    def __init__(self) -> None:
        # off until we know the keys exist and the client starts
        self.enabled = False
        self._lf = None
        # no keys in .env: skip Langfuse completely, the bot still works
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            logger.info("Langfuse keys not set - tracing disabled")
            return
        try:
            # imported here so the project runs even if the langfuse package has a problem
            from langfuse import get_client
            self._lf = get_client()
            self.enabled = True
        except Exception:  # pragma: no cover - defensive
            # anything going wrong with Langfuse must not stop the chatbot
            logger.warning("Langfuse init failed - tracing disabled", exc_info=True)

    @contextmanager
    def turn(self, session_id: str, turn_id: int, user_query: str) -> Iterator[Any]:
        """One Langfuse trace per chat turn; all LangChain calls inside nest beneath it."""
        # tracing off: hand out the do-nothing span and carry on
        if not self.enabled:
            yield _NullSpan()
            return
        stack = ExitStack()
        try:
            from langfuse import propagate_attributes
            # session_id groups the turns of one chat in the Langfuse Sessions tab
            stack.enter_context(propagate_attributes(session_id=session_id, trace_name="chat_turn",
                                                     metadata={"turn_id": str(turn_id)},
                                                     tags=["unicorn-chatbot"]))
            # all model calls started inside this span appear beneath the chat turn
            span = stack.enter_context(self._lf.start_as_current_observation(
                name="chat_turn", as_type="span", input=user_query))
        except Exception:
            self._close_safely(stack)
            logger.warning("Langfuse trace start failed - continuing untraced", exc_info=True)
            yield _NullSpan()
            return
        try:
            yield span
        except BaseException:
            # Preserve the application's exception while still closing the trace context.
            self._close_safely(stack, *sys.exc_info())
            raise
        else:
            self._close_safely(stack)

    @contextmanager
    def step(self, name: str, **input_: Any) -> Iterator[Any]:
        """A child span for a pipeline stage (sanitize / retrieve / ...)."""
        if not self.enabled:
            yield _NullSpan()
            return
        # same idea as turn(), but for a smaller step inside the turn
        stack = ExitStack()
        try:
            span = stack.enter_context(self._lf.start_as_current_observation(
                name=name, as_type="span", input=input_ or None))
        except Exception:
            self._close_safely(stack)
            logger.warning("Langfuse span start failed - continuing without this span", exc_info=True)
            yield _NullSpan()
            return
        try:
            yield span
        except BaseException:
            self._close_safely(stack, *sys.exc_info())
            raise
        else:
            self._close_safely(stack)

    @staticmethod
    def _close_safely(stack: ExitStack, *exc_info: Any) -> None:
        # A tracing cleanup error should never replace the chatbot's own result or error.
        try:
            if exc_info:
                stack.__exit__(*exc_info)
            else:
                stack.close()
        except Exception:
            logger.warning("Langfuse context close failed - continuing", exc_info=True)

    def callbacks(self) -> list[Any]:
        """LangChain callback list so LLM/embedding calls (tokens, cost, latency) show up in Langfuse."""
        if not self.enabled:
            return []
        # this is what makes the LLM calls appear nested inside our trace, with tokens and cost
        try:
            from langfuse.langchain import CallbackHandler
            return [CallbackHandler()]
        except Exception:  # pragma: no cover - defensive
            return []

    def update_span(self, span: Any, **values: Any) -> None:
        # All span writes go through here so a telemetry failure cannot interrupt a chat turn.
        try:
            span.update(**values)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Langfuse span update failed", exc_info=True)

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        # attach a number to the current trace (we use 0 or 1 for clarification and fallback)
        if not self.enabled:
            return
        try:
            self._lf.score_current_trace(name=name, value=float(value), data_type="NUMERIC", comment=comment)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Langfuse score failed", exc_info=True)

    def flush(self) -> None:
        # Langfuse sends data in the background; flush at exit so the last traces aren't lost
        if self.enabled:
            try:
                self._lf.flush()
            except Exception:  # pragma: no cover - defensive
                pass
