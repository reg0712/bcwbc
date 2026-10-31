"""Model, tool and retrieval backends, bundled for the executor."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any
from pathlib import Path
from uuid import uuid4

import httpx

from synapse.backends.llm import LLMAdapter, MockLLM, live_llm
from synapse.backends.retrieval import EmbeddingRetriever, Retriever
from synapse.backends.tools import ToolClient
from synapse.config import load_config, load_env


@dataclass
class Backends:
    """What a run calls out to: an LLM adapter, the bank API tools and a retriever, plus the
    provider that prices the LLM calls and its concurrency cap."""

    llm: LLMAdapter
    tools: ToolClient
    retriever: Retriever
    functions: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = field(default_factory=dict)
    provider: str = "mock"
    max_concurrency: int = 8
    latency_scale: float = 1.0
    model_names: dict[str, str] = field(default_factory=dict)
    cache_namespace: str = field(default_factory=lambda: uuid4().hex)


@asynccontextmanager
async def mock_backends(scale: float | None = None, *, bank_url: str | None = None,
                        bank_app: Any = None,
                        retriever: Retriever | None = None,
                        functions: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None
                        ) -> AsyncIterator[Backends]:
    """Mock-mode backends: the mock LLM, the bank API (in-process unless bank_url is given)
    and local retrieval, every delay multiplied by scale (config/mock.yaml by default)."""
    from mock_services import MOCK_TASKS
    from mock_services.bank_api import create_app

    mock = load_config("mock.yaml")
    scale = mock.get("latency_scale", 1.0) if scale is None else scale
    tiers = load_config("models.yaml")["providers"]["mock"]
    llm = MockLLM(MOCK_TASKS, mock["llm_latency_ms"],
                  {"small": tiers["small"], "large": tiers["large"]}, scale)
    if retriever is None:
        retriever = EmbeddingRetriever(extra_latency_ms=mock["retrieve_latency_ms"] * scale)
    if bank_url:
        client = httpx.AsyncClient(base_url=bank_url, timeout=30)
    else:
        app = bank_app if bank_app is not None else create_app(scale=scale)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                   base_url="http://bank.mock", timeout=30)
    async with client:
        yield Backends(llm=llm, tools=ToolClient(client), retriever=retriever,
                       functions=functions or {},
                       model_names={"small": tiers["small"], "large": tiers["large"]},
                       latency_scale=scale,
                       max_concurrency=int(tiers.get("max_concurrency", 8)))


@asynccontextmanager
async def live_backends(scale: float | None = None, *, provider: str | None = None,
                        model: str | None = None,
                        quota_path: str | Path | None = None,
                        llm: LLMAdapter | None = None,
                        **options: Any) -> AsyncIterator[Backends]:
    """Live-mode backends: real LLM calls to the provider in config/models.yaml (key from
    .env); the bank APIs and retrieval stay local, with their simulated delays."""
    load_env()
    models = load_config("models.yaml")
    provider = provider or models["live_provider"]
    if provider not in ("gemini", "groq"):
        raise ValueError(f"unsupported live provider {provider!r}")
    settings = models["providers"][provider]
    if model is not None and model not in (settings["small"], settings["large"]):
        raise ValueError(f"{model!r} is not a configured {provider} model")
    adapter = llm if llm is not None else live_llm(provider, settings, model, quota_path)
    async with mock_backends(scale, **options) as backends:
        backends.llm = adapter
        backends.provider = provider
        backends.model_names = ({"small": model, "large": model} if model else
                                {"small": settings["small"], "large": settings["large"]})
        backends.max_concurrency = int(settings.get("max_concurrency", 4))
        yield backends


@asynccontextmanager
async def open_backends(live: bool = False, scale: float | None = None,
                        *, provider: str | None = None,
                        model: str | None = None,
                        **options: Any) -> AsyncIterator[Backends]:
    """Backends for a run. Mock by default; with live=True, LLM calls go to the provider in
    config/models.yaml while bank APIs and retrieval stay local. options pass through to
    mock_backends (bank_url, bank_app, retriever, functions)."""
    if not live and provider not in (None, "mock"):
        raise ValueError("a live provider requires live=True")
    if not live and model is not None:
        raise ValueError("a live model requires live=True")
    factory = live_backends if live else mock_backends
    settings = {"provider": provider, "model": model} if live else {}
    async with factory(scale, **settings, **options) as backends:
        yield backends
