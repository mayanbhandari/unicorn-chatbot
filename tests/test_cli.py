#Test the CLI demo's retry and transcript behavior without calling a live model provider. These fakes let the tests check each outcome quickly, without waiting through real retry delays.

from types import SimpleNamespace

from src import cli
from src.chatbot import BUSY_MESSAGE, DAILY_QUOTA_MESSAGE


class FakeMetrics:
    # run_demo prints a metrics summary, so the fake bot needs this small stand-in.
    def snapshot(self):
        return {"queries": 2, "errors": 1}


class SequencedBot:
    # Return prepared replies in order, letting each test control whether a retry succeeds.
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0
        self.metrics = FakeMetrics()

    def handle(self, session, query):
        # Count each attempt so tests can verify the retry limit.
        self.calls += 1
        return next(self.replies)


def _reply(text, *, error=False, blocked=False):
    # The demo only needs these reply fields; a SimpleNamespace keeps the fake lightweight.
    return SimpleNamespace(text=text, error=error, blocked=blocked, intent="error" if error else "filter_list",
                           clarification=False, latency_ms=10)


def test_demo_retries_busy_response_and_prints_success(monkeypatch, capsys):
    # Simulate one temporary provider error followed by a successful answer.
    bot = SequencedBot([
        _reply(BUSY_MESSAGE, error=True),
        _reply("Found 1 company: Flipkart"),
    ])
    sleeps = []
    # Keep the demo to one prompt and capture the retry delay instead of actually waiting.
    monkeypatch.setattr(cli, "DEMO_SESSIONS", [("retry case", ["List fintech unicorns"])])
    monkeypatch.setattr(cli, "DEMO_PAUSE_S", 0)
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    cli.run_demo(bot)

    # It retries once, waits once, prints the final answer, and reports progress on stderr.
    output = capsys.readouterr()
    assert bot.calls == 2
    assert sleeps == [cli.DEMO_RETRY_WAIT_S]
    assert "**Bot:** Found 1 company: Flipkart" in output.out
    assert "provider busy" in output.err


def test_demo_reports_final_busy_response_after_retry_limit(monkeypatch, capsys):
    # Keep every attempt busy to confirm the demo stops after the initial call plus two retries.
    bot = SequencedBot([_reply(BUSY_MESSAGE, error=True) for _ in range(cli.DEMO_RETRIES + 1)])
    sleeps = []
    # Avoid the real retry and demo pauses so this remains a fast offline test.
    monkeypatch.setattr(cli, "DEMO_SESSIONS", [("retry case", ["List fintech unicorns"])])
    monkeypatch.setattr(cli, "DEMO_PAUSE_S", 0)
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    cli.run_demo(bot)

    # The retry cap is respected, and the last busy response is what the transcript displays.
    output = capsys.readouterr()
    assert bot.calls == cli.DEMO_RETRIES + 1
    assert sleeps == [cli.DEMO_RETRY_WAIT_S] * cli.DEMO_RETRIES
    assert f"**Bot:** {BUSY_MESSAGE}" in output.out


def test_demo_does_not_retry_daily_quota_exhaustion(monkeypatch, capsys):
    # A daily quota will not recover after 40 seconds, so the demo should stop immediately.
    bot = SequencedBot([_reply(DAILY_QUOTA_MESSAGE, error=True)])
    sleeps = []
    # Use one prompt and record any sleep calls to prove no retry delay was started.
    monkeypatch.setattr(cli, "DEMO_SESSIONS", [("quota case", ["List fintech unicorns"])])
    monkeypatch.setattr(cli, "DEMO_PAUSE_S", 0)
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    cli.run_demo(bot)

    # The daily-quota message is shown to the user after just one attempt.
    output = capsys.readouterr()
    assert bot.calls == 1
    assert sleeps == []
    assert f"**Bot:** {DAILY_QUOTA_MESSAGE}" in output.out
