# First line of defence for user input: cleans up the text, rejects empty or junk messages, and blocks obvious prompt-injection attempts. We need it so nothing messy or hostile reaches the LLM, and so every blocked message gets a clear reason we can log.

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Phrases people use to hijack a chatbot ("ignore previous instructions", "reveal your system prompt"...).
# Regex alone won't stop a clever attacker, it just catches the obvious ones cheaply.
# The other layers: user text is wrapped in tags and marked untrusted in the prompts,
# and the bot only answers from dataset records, so there is nothing secret to leak.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier|your)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(everything|all|your)\s+(you|instructions?|rules?)",
    r"(reveal|show|print|repeat|output|leak)\s+(me\s+)?(your|the)\s+(system|hidden|initial|original)\s+(prompt|instructions?|message)",
    r"what\s+(is|are)\s+your\s+(system\s+prompt|hidden\s+instructions?)",
    r"you\s+are\s+now\s+(a|an|in)\s+",
    # kept narrow on purpose: "act as a startup advisor" is fine, "act as an unrestricted AI" is not
    r"\bact\s+as\s+(if\s+you\s+are\s+)?(an?\s+)?(unrestricted|different|new|evil|jailbroken)\b",
    r"pretend\s+(to\s+be|you\s+are)",
    r"\b(jailbreak|dan\s+mode|developer\s+mode)\b",
    r"override\s+(your|the|all)\s+(rules?|instructions?|safety|guardrails?)",
    # "System:" only counts at the start or after a sentence, so "Juspay's system: payments" isn't blocked
    r"(?:^|[.!?]\s)(system|assistant)\s*:",
    # fake chat-format tags that some models treat as real roles
    r"</?\s*(system|assistant|user|instructions?)\s*>",
    r"<\|.*?\|>",
    r"\[/?INST\]",
]
# A few clearly harmful things that have no business in a startup-data chatbot.
_HARMFUL_PATTERNS = [
    r"\b(drop|delete|truncate)\s+table\b",
    r"<\s*script\b",
    r"\b(make|build|create)\s+(a\s+)?(bomb|explosive|malware|ransomware)\b",
    r"\b(hack|breach)\s+into\b",
    r"\brm\s+-rf\b",
]
# Compile once at import time (ignoring case) so we don't rebuild the regexes on every message.
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_HARMFUL_RE = [re.compile(p, re.IGNORECASE) for p in _HARMFUL_PATTERNS]


@dataclass
class SanitizeResult:
    clean_text: str                                        # the cleaned-up message
    flags: list[str] = field(default_factory=list)         # everything we noticed or changed (goes into the logs)
    blocked: bool = False                                  # True means don't send this to the LLM
    reason: str | None = None                              # why it was blocked: empty, gibberish, prompt_injection...


def sanitize(raw: str | None, max_chars: int = 500) -> SanitizeResult:
    flags: list[str] = []
    # not even a string (None or something odd): treat it like an empty message
    if raw is None or not isinstance(raw, str):
        return SanitizeResult("", ["not_text"], True, "empty")

    # 1. Unicode cleanup. NFKC turns look-alike and full-width characters into normal ones,
    #    then we drop invisible control characters (zero-width spaces and similar) but keep real line breaks as spaces.
    text = unicodedata.normalize("NFKC", raw)
    cleaned = "".join(" " if ch in "\n\r\t" else ch for ch in text
                      if ch in "\n\r\t" or unicodedata.category(ch)[0] != "C")
    # if the result differs from the input (ignoring whitespace swaps), something sneaky was removed, so note it
    if cleaned != text.replace("\n", " ").replace("\r", " ").replace("\t", " "):
        flags.append("control_chars_removed")

    # 2. Squash repeated spaces into one and trim the ends
    text = re.sub(r"\s+", " ", cleaned).strip()
    if not text:
        return SanitizeResult("", flags + ["empty"], True, "empty")

    # 3. Cut very long messages instead of rejecting them
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
        flags.append("truncated")

    # 4. Junk check: if there are hardly any letters or digits (like "?!?!?!" or a wall of emoji), block it
    alnum = sum(ch.isalnum() for ch in text)
    if alnum < 2 or alnum / len(text) < 0.4:
        return SanitizeResult(text, flags + ["gibberish"], True, "gibberish")

    # 5. Run the injection patterns first, then the harmful ones; the first match blocks the message
    for rx in _INJECTION_RE:
        if rx.search(text):
            return SanitizeResult(text, flags + ["prompt_injection"], True, "prompt_injection")
    for rx in _HARMFUL_RE:
        if rx.search(text):
            return SanitizeResult(text, flags + ["harmful_pattern"], True, "harmful_pattern")

    # 6. Our prompts wrap the question in <user_query> tags, so remove those tags if the user types them.
    #    Otherwise they could "close" the block early and sneak text outside it.
    if re.search(r"</?user_query>", text, re.IGNORECASE):
        text = re.sub(r"</?user_query>", "", text, flags=re.IGNORECASE).strip()
        flags.append("delimiter_stripped")

    # all good, hand back the cleaned text
    return SanitizeResult(text, flags)