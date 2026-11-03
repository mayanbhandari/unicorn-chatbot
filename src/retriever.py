#Semantic retrieval: Finds companies by meaning instead of exact words, using embeddings stored in a Chroma vector database. We need it for vague questions like "a good payments partner for my shop", where filters on columns can't help.

from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Optional, Protocol

import chromadb
import pandas as pd
from chromadb.config import Settings as ChromaSettings


class Embedder(Protocol):
    # Anything with these two methods works as an embedder (Gemini, OpenAI, Ollama, or the fake one in tests).
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class CompanyRetriever:
    def __init__(self, df: pd.DataFrame, embedder: Embedder, persist_dir: Path, model_name: str,
                 batch_size: int = 90, batch_pause_s: float = 61.0) -> None:
        self.df = df
        self.embedder = embedder
        # Free API tiers cap embedding requests per minute (Gemini: 100/min), so indexing is throttled.
        self.batch_size = batch_size
        self.batch_pause_s = batch_pause_s
        # the index is saved on disk, so we only pay the embedding cost on the very first run
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir),
                                                 settings=ChromaSettings(anonymized_telemetry=False))
        # fingerprint of (model + document text) => stale indexes are never reused
        fp = hashlib.sha1((model_name + "".join(df["doc_text"])).encode()).hexdigest()[:10]
        # collection names only allow simple characters, so clean up the model name first
        safe_model = re.sub(r"[^a-zA-Z0-9]+", "_", model_name)[:30]
        # the name includes the fingerprint, so changing the model or the data automatically gives a fresh collection
        self._collection = self._client.get_or_create_collection(
            name=f"unicorns_{safe_model}_{fp}"[:60], metadata={"hnsw:space": "cosine"})
        # empty or incomplete? build the index now
        if self._collection.count() != len(df):
            self._index()

    def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """Embed in small batches, pausing between them and retrying on rate-limit (429) errors."""
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            chunk = texts[start:start + self.batch_size]
            # printed to stderr so the user can see progress during the one-time wait
            print(f"Building embedding index (first run only): companies {start + 1}-{start + len(chunk)} of {total}",
                  file=sys.stderr, flush=True)
            # up to 5 tries per batch
            for attempt in range(1, 6):
                try:
                    vectors.extend(self.embedder.embed_documents(chunk))
                    break
                except Exception as exc:  # rate limit -> wait and retry; anything else is a real error
                    rate_limited = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                    # a wrong key or a bug shouldn't be retried, and neither should the 5th failure
                    if not rate_limited or attempt == 5:
                        raise
                    print(f"Rate limited by the embedding API; waiting 60s (attempt {attempt}/5)...",
                          file=sys.stderr, flush=True)
                    time.sleep(60)
            # pause between batches so we stay under the per-minute limit (no pause after the last one)
            if start + self.batch_size < total:
                time.sleep(self.batch_pause_s)
        return vectors

    def _index(self) -> None:
        # embed the descriptive text of every company
        texts = self.df["doc_text"].tolist()
        embeddings = self._embed_in_batches(texts)        # nothing is written unless ALL batches succeed
        # save text, vector and name together, keyed by company id
        self._collection.upsert(
            ids=[str(i) for i in self.df["id"]],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"company": c} for c in self.df["company"]],
        )

    def search(self, query: str, k: int = 5, allowed_ids: Optional[list[int]] = None) -> list[tuple[int, float]]:
        """Return [(company_id, similarity)] best first, optionally restricted to a candidate set."""
        n = len(self.df)
        # embed the question and ask Chroma for everything ranked by closeness (only 118 rows, so asking for all is cheap)
        res = self._collection.query(query_embeddings=[self.embedder.embed_query(query)], n_results=n)
        # a follow-up or a filter may limit us to certain companies
        allowed = set(allowed_ids) if allowed_ids is not None else None
        hits: list[tuple[int, float]] = []
        for id_str, dist in zip(res["ids"][0], res["distances"][0]):
            cid = int(id_str)
            # the results are already best-first, so keep the first k that are allowed
            if allowed is None or cid in allowed:
                hits.append((cid, round(1.0 - float(dist), 4)))   # cosine distance -> similarity
            if len(hits) >= k:
                break
        return hits