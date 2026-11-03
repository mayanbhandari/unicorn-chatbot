# These tests keep Langfuse optional in practice: telemetry startup and cleanup problems must not break a chat turn.

from contextlib import contextmanager
import sys
import types

from src.metrics import Observability


@contextmanager
def _raising_context(*args, **kwargs):
    raise RuntimeError("simulated Langfuse failure")
    yield  # pragma: no cover - lets this function be recognized as a context manager


def test_turn_continues_if_langfuse_setup_fails(monkeypatch):
    # Simulate a failure while entering the Langfuse trace context.
    langfuse = types.ModuleType("langfuse")
    langfuse.propagate_attributes = _raising_context
    monkeypatch.setitem(sys.modules, "langfuse", langfuse)
    obs = Observability()
    obs.enabled = True
    obs._lf = object()

    with obs.turn("session-1", 1, "hello") as span:
        span.update(output="chat still works")


def test_step_continues_if_langfuse_span_setup_fails():
    # A child span failing should only remove that trace detail, not interrupt its caller.
    class BrokenClient:
        def start_as_current_observation(self, **kwargs):
            return _raising_context()

    obs = Observability()
    obs.enabled = True
    obs._lf = BrokenClient()

    with obs.step("route", query="hello") as span:
        span.update(output="route still works")


def test_trace_cleanup_failure_is_suppressed():
    # Context-manager cleanup happens after the bot has a reply; a cleanup outage must not discard it.
    @contextmanager
    def context_that_fails_on_close():
        yield object()
        raise RuntimeError("simulated Langfuse cleanup failure")

    class BrokenClient:
        def start_as_current_observation(self, **kwargs):
            return context_that_fails_on_close()

    obs = Observability()
    obs.enabled = True
    obs._lf = BrokenClient()

    with obs.step("route"):
        pass


def test_score_failure_is_suppressed():
    # A failed score upload must never stop the chatbot from replying.
    class BrokenClient:
        def score_current_trace(self, **kwargs):
            raise RuntimeError("simulated Langfuse score failure")

    obs = Observability()
    obs.enabled = True
    obs._lf = BrokenClient()

    obs.score("fallback", 1)
