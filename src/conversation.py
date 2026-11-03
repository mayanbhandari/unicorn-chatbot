# Holds everything the bot remembers about one chat: recent messages, the companies it last listed, and how many questions it has asked. We need it so follow-ups like "which of these are in Bangalore?" can work without the LLM having to remember anything.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Session:
    # short random id so every chat can be traced in the logs and in Langfuse
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # I keep two kinds of memory. First: the recent messages, so the LLM can work out what "it" or "those" means.
    history: list[tuple[str, str]] = field(default_factory=list)      # (role, text)
    # Second: the exact company ids from the last answer. "Which of these..." is then filtered
    # against real ids, instead of trusting the LLM to remember a list correctly.
    last_result_ids: list[int] = field(default_factory=list)
    # how many clarifying questions in a row we've asked (used to stop endless question loops)
    clarification_streak: int = 0
    # how many messages the user has sent in this chat
    turn_count: int = 0

    def add(self, role: str, text: str, max_chars: int = 700) -> None:
        # long answers (like a 29-company list) are cut short in the history so prompts stay small
        self.history.append((role, text if len(text) <= max_chars else text[:max_chars] + " …"))

    def recent_history(self, turns: int) -> list[tuple[str, str]]:
        # one turn = a user message plus a bot reply, so we take 2 messages per turn from the end
        return self.history[-2 * turns:]

    def history_text(self, turns: int) -> str:
        msgs = self.recent_history(turns)
        # plain "USER: ... / ASSISTANT: ..." text that can be pasted straight into a prompt
        return "\n".join(f"{role.upper()}: {text}" for role, text in msgs) or "(no previous messages)"

    def previous_results_text(self, df: pd.DataFrame, limit: int = 40) -> str:
        # nothing shown yet, so there is nothing for "these" to point at
        if not self.last_result_ids:
            return "(none)"
        # turn the saved ids back into company names (skipping any id that isn't in the data)
        names = df.set_index("id").loc[[i for i in self.last_result_ids if i in set(df["id"])], "company"].tolist()
        # numbered list, so the router can understand "the second one"
        shown = "; ".join(f"{i + 1}. {n}" for i, n in enumerate(names[:limit]))
        # if the list was long, say how many we left out
        return shown + (f"; ... (+{len(names) - limit} more)" if len(names) > limit else "")

    def reset(self) -> None:
        # used by the /reset command in the CLI: forget everything and start fresh
        self.history.clear()
        self.last_result_ids.clear()
        self.clarification_streak = 0
        self.turn_count = 0