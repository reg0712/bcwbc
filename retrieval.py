"""Local retrieval over small JSON indexes in data/ (data/<index>.json), on CPU.

EmbeddingRetriever ranks passages with a sentence-transformers model. KeywordRetriever ranks
them by shared words and needs no model, for tests.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Iterable
from typing import Any, Protocol

from synapse import ROOT
from synapse.config import load_config

DATA = ROOT / "data"
_WORD = re.compile(r"[a-z0-9]+")


class Retriever(Protocol):
    async def search(self, index: str, query: str, k: int = 3) -> list[dict[str, Any]]: ...

    async def warm(self, indexes: Iterable[str]) -> None: ...


def load_passages(index: str) -> list[dict[str, Any]]:
    """The passages of a named index; each has an id, a title and text."""
    return json.loads((DATA / f"{index}.json").read_text())["passages"]


def embedding_model_name() -> str:
    """The sentence-transformers model named in config/models.yaml."""
    return load_config("models.yaml")["embedding_model"]


def download_model() -> None:
    """Fetch the embedding model into the local cache, so later runs work offline."""
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(embedding_model_name(), device="cpu")


class EmbeddingRetriever:
    """Cosine similarity over sentence-transformers embeddings. In mock mode,
    extra_latency_ms adds the round trip a remote vector store would take."""

    def __init__(self, model_name: str | None = None, extra_latency_ms: float = 0.0) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, deferred
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()  # keeps CLI output clean
        name = model_name or embedding_model_name()
        try:  # a cached model loads without contacting the Hub, so runs work offline
            self._model = SentenceTransformer(name, device="cpu", local_files_only=True)
        except OSError:
            self._model = SentenceTransformer(name, device="cpu")
        self._extra_s = extra_latency_ms / 1000
        self._indexes: dict[str, tuple[list[dict[str, Any]], Any]] = {}
        self._lock = threading.Lock()

    async def warm(self, indexes: Iterable[str]) -> None:
        """Embed the named indexes now, so no run pays for it."""
        for index in indexes:
            await asyncio.to_thread(self._index, index)

    async def search(self, index: str, query: str, k: int = 3) -> list[dict[str, Any]]:
        """The k passages closest to the query, best first, each with its score."""
        await asyncio.sleep(self._extra_s)
        return await asyncio.to_thread(self._search, index, query, k)

    def _search(self, index: str, query: str, k: int) -> list[dict[str, Any]]:
        passages, vectors = self._index(index)
        with self._lock:
            query_vector = self._model.encode([query], normalize_embeddings=True)[0]
        scores = vectors @ query_vector
        ranked = sorted(range(len(passages)), key=lambda i: (-scores[i], i))[:k]
        return [{**passages[i], "score": round(float(scores[i]), 4)} for i in ranked]

    def _index(self, index: str) -> tuple[list[dict[str, Any]], Any]:
        with self._lock:
            if index not in self._indexes:
                passages = load_passages(index)
                texts = [f"{p['title']}. {p['text']}" for p in passages]
                self._indexes[index] = (passages, self._model.encode(texts, normalize_embeddings=True))
            return self._indexes[index]


class KeywordRetriever:
    """Ranks passages by how many query words they contain; deterministic, no model."""

    async def warm(self, indexes: Iterable[str]) -> None:
        """Nothing to prepare."""

    async def search(self, index: str, query: str, k: int = 3) -> list[dict[str, Any]]:
        """The k passages sharing the most words with the query, ties in index order."""
        words = set(_WORD.findall(query.lower()))
        passages = load_passages(index)
        overlap = [len(words & set(_WORD.findall(f"{p['title']} {p['text']}".lower())))
                   for p in passages]
        ranked = sorted(range(len(passages)), key=lambda i: (-overlap[i], i))[:k]
        return [{**passages[i], "score": float(overlap[i])} for i in ranked]
