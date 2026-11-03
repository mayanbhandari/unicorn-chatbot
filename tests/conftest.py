# Shared test setup: loads the real dataset, plus fake stand-ins for the LLM and embeddings so tests run offline. We need it so the whole bot can be tested quickly and the same way every time, with no API keys and no internet.
import hashlib                          # to turn words into stable numbers for the fake embedder
import re
from types import SimpleNamespace       # a quick object with attributes, used to mimic an LLM reply

import pytest

from src.chatbot import Chatbot
from src.config import Settings
from src.conversation import Session
from src.data_loader import load_companies
from src.metrics import Metrics, Observability
from src.retriever import CompanyRetriever
from src.router import Route, Router
from src.structured_query import Filters


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic and good enough to test retrieval plumbing."""
    # Not real AI. It just counts words, so the same text always gives the same vector and no API is needed.
    DIM = 256

    def _vec(self, text):
        v = [0.0] * self.DIM
        # split the text into lowercase words
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            # hash each word to a slot in the vector and add one there
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1.0
        return v

    # the two methods the retriever expects from any embedder
    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class ScriptedRouterLLM:
    """Stands in for the structured-output LLM: a few keyword rules that mimic what the real router returns."""
    # Instead of asking a real model, it looks for keywords in the question and returns the Route a good model would.

    def __init__(self):
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        # the second message is the user part of the prompt; pull the question out of the <user_query> tags
        human = messages[1][1]
        q = re.search(r"<user_query>(.*?)</user_query>", human, re.S).group(1).lower()
        # the word "boom" triggers a fake crash, used to test error handling
        if "boom" in q:
            raise RuntimeError("simulated LLM outage")
        # one rule per behaviour we want to test, checked from most specific to least
        if q in ("hi", "hello"):
            return Route(intent="chitchat")
        if "weather" in q:
            return Route(intent="out_of_scope")
        if "razorpay" in q or "razor pay" in q:
            return Route(intent="company_info", company_names=["Razorpay"])
        if "acmecorp" in q:
            return Route(intent="company_info", company_names=["AcmeCorp"])
        if "fintech" in q:
            return Route(intent="filter_list", filters=Filters(sector_groups=["Fintech"]))
        # a follow-up that points at the previous list
        if "bangalore" in q and "these" in q:
            return Route(intent="filter_list", uses_previous_results=True, filters=Filters(locations=["Bangalore"]))
        if "most valuable" in q:
            return Route(intent="filter_list", filters=Filters(sort_by="valuation", limit=1))
        # a vague "collaborate" question with no detail makes the router ask a clarifying question
        if "collaborate" in q and "payments" not in q and "ecommerce" not in q and "e-commerce" not in q:
            return Route(intent="recommend", needs_clarification=True,
                         clarifying_question="Can you clarify the type of collaboration (e.g., technology, payments, logistics)?")
        # once the user adds detail, it answers with a recommendation
        if "payments" in q or "collaborate" in q:
            return Route(intent="recommend", semantic_query="payments gateway for e-commerce businesses")
        # anything else counts as off-topic
        return Route(intent="out_of_scope")


class FakeAnswerLLM:
    # Fake answer writer. It remembers the last prompt it received, so tests can check that the records were really included.
    def __init__(self):
        self.last_prompt = None

    def invoke(self, messages, config=None):
        self.last_prompt = messages[1][1]
        return SimpleNamespace(content="[fake answer grounded in records]")


# A "fixture" is setup code that pytest hands to any test that asks for it by name.
# scope="session" means it is created once and shared by all tests.
@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    # use a temporary folder, so tests never touch your real logs or your real embedding index
    base = tmp_path_factory.mktemp("work")
    return Settings(chroma_dir=base / "chroma", log_dir=base / "logs")


# the real dataset, loaded once
@pytest.fixture(scope="session")
def df(settings):
    return load_companies(settings.data_path)


# a fresh bot for every test, built from the real data and the fakes above
@pytest.fixture()
def bot(df, settings, monkeypatch):
    from src.logging_setup import setup_logging
    # Tests must remain deterministic and offline even when a developer's shell
    # or .env file contains live Langfuse credentials.
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    # write logs to the temporary folder
    setup_logging(settings.log_dir)
    # batch_pause_s=0 so the test doesn't wait the 61 seconds the real embedding limit needs
    retriever = CompanyRetriever(df, FakeEmbedder(), settings.chroma_dir, "fake-embedder", batch_pause_s=0)
    # the real Router and Chatbot code, with only the LLMs replaced by fakes
    router = Router(ScriptedRouterLLM(), df, settings)
    b = Chatbot(df, router, FakeAnswerLLM(), retriever, settings, Metrics(), Observability())
    return b


# a clean conversation for each test
@pytest.fixture()
def session():
    return Session()