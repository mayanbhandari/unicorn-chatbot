# Offline tests that the right LLM classes get built for each provider, and that embeddings are sent in paced batches. We need them to prove the Gemini / Ollama / OpenAI switch and the free-tier rate-limit fix work without calling any real API.

from __future__ import annotations

import sys
import types
from dataclasses import replace

import pytest

from src.chatbot import build_models
from src.retriever import CompanyRetriever


# A fake chat model that just records how it was created and how structured output was requested.
class FakeChatModel:
    def __init__(self, **kwargs):
        # keep the settings we were built with, so the test can check them
        self.kwargs = kwargs
        self.structured_output = None

    def with_structured_output(self, schema, **kwargs):
        # remember what was asked for (for example method="function_calling" on OpenAI)
        self.structured_output = (schema, kwargs)
        return ("structured", schema, kwargs)


class FakeEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


# parametrize runs the same test three times, once per provider
@pytest.mark.parametrize("provider,module_name,chat_class,embedding_class,expected_method", [
    ("gemini", "langchain_google_genai", "ChatGoogleGenerativeAI",
     "GoogleGenerativeAIEmbeddings", None),
    ("ollama", "langchain_ollama", "ChatOllama", "OllamaEmbeddings", None),
    ("openai", "langchain_openai", "ChatOpenAI", "OpenAIEmbeddings", "function_calling"),
])
def test_build_models_wires_each_provider(
    monkeypatch, settings, provider, module_name, chat_class, embedding_class, expected_method
):
    # Build a pretend provider package that contains our fake classes, and register it as if it were installed.
    # That way build_models() imports the fakes and never touches the real libraries or the internet.
    module = types.ModuleType(module_name)
    setattr(module, chat_class, FakeChatModel)
    setattr(module, embedding_class, FakeEmbedder)
    monkeypatch.setitem(sys.modules, module_name, module)
    # copy the settings with the provider and model names changed
    configured = replace(settings, llm_provider=provider, chat_model="test-chat",
                          embedding_model="test-embedding")

    llm, router_llm, embedder = build_models(configured)

    # the chosen model names were passed through to the right classes
    assert llm.kwargs["model"] == "test-chat"
    assert embedder.kwargs["model"] == "test-embedding"
    # the router LLM was set up for structured output
    assert router_llm[0] == "structured"
    # OpenAI should use the "function_calling" method; the others use the default
    if expected_method:
        assert router_llm[2]["method"] == expected_method


# This protects the free-tier fix: 118 texts must go out in batches, with a pause between batches.
def test_embedding_index_batches_and_pauses_between_batches(monkeypatch):
    # a fake embedder that only records how many texts each call received
    class RecordingEmbedder:
        def __init__(self):
            self.batch_sizes = []

        def embed_documents(self, texts):
            self.batch_sizes.append(len(texts))
            return [[float(len(texts))] for _ in texts]

    # replace time.sleep so the test runs instantly and just records the pauses it would have taken
    sleeps = []
    monkeypatch.setattr("src.retriever.time.sleep", sleeps.append)
    # build a retriever without running its __init__ (no Chroma needed); we only test the batching method
    retriever = object.__new__(CompanyRetriever)
    retriever.embedder = RecordingEmbedder()
    retriever.batch_size = 50
    retriever.batch_pause_s = 61.0

    # 118 fake company texts
    vectors = retriever._embed_in_batches([f"company {i}" for i in range(118)])

    # 50 + 50 + 18 = 118, so three calls
    assert retriever.embedder.batch_sizes == [50, 50, 18]
    # one vector per text, nothing lost
    assert len(vectors) == 118
    # a pause between batches, but none after the last one (so two pauses for three batches)
    assert sleeps == [61.0, 61.0]
