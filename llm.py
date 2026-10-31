"""LLM adapters behind one interface: a deterministic mock, plus Groq and Gemini through
LangChain chat models at temperature 0. LangChain is imported here and nowhere else.

Prompts are templates under prompts/. The executor renders a node's inputs as one JSON block
inside <inputs> tags, placed at the template's {inputs} slot (or appended), so static
instructions come first. A node whose tests name a schema (`schema: risk.json`) is structured:
its output must match that JSON Schema file under prompts/.

MockLLM answers from deterministic task functions; LangChainLLM calls the live provider
named in config/models.yaml (Gemini or Groq).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from synapse import ROOT
from synapse.backends.quota import QuotaExceeded, QuotaGuard
from synapse.backends.encoding import TABLE_HINT, encode_inputs, unpack
from synapse.ir import Node, Usage

PROMPTS = ROOT / "prompts"
_INPUTS_SLOT = "{inputs}"
_INPUTS_BLOCK = re.compile(r'<inputs(?P<tables> format="tables")?>\n(?P<data>.*)\n</inputs>', re.DOTALL)
LLM_EVENTS: ContextVar[Any] = ContextVar("llm_events", default=None)
_SCALARS: dict[str, type] = {"string": str, "integer": int, "number": float, "boolean": bool}


class InvalidStructuredOutput(ValueError):
    def __init__(self, message: str, usage: Usage) -> None:
        super().__init__(message)
        self.usage = usage


class LiveProviderError(RuntimeError):
    """A short live-model failure safe to display on the dashboard."""


class LLMAdapter(Protocol):
    async def invoke(self, node: Node, prompt: str,
                     schema: type[BaseModel] | None) -> tuple[dict[str, Any], Usage]: ...


def render_prompt(node: Node, inputs: Mapping[str, Any]) -> str:
    """The node's prompt template with its inputs filled in as a JSON block."""
    template = (ROOT / str(node.prompt)).read_text()
    if node.compact_inputs:
        data, tables = encode_inputs(dict(inputs))
        tag = '<inputs format="tables">' if tables else '<inputs>'
        block = f"{TABLE_HINT if tables else ''}{tag}\n{data}\n</inputs>"
    else:
        block = f"<inputs>\n{json.dumps(dict(inputs), ensure_ascii=False)}\n</inputs>"
    if _INPUTS_SLOT in template:
        return template.replace(_INPUTS_SLOT, block)
    return f"{template.rstrip()}\n\n{block}\n"


def extract_inputs(prompt: str) -> dict[str, Any]:
    """The inputs block of a rendered prompt, parsed back into a dict."""
    match = _INPUTS_BLOCK.search(prompt)
    if match is None:
        raise ValueError("prompt has no <inputs> block")
    value = json.loads(match.group("data"))
    return unpack(value) if match.group("tables") else value


def estimate_tokens(text: str) -> int:
    """Rough token count: characters divided by 4, rounded up."""
    return math.ceil(len(text) / 4)


def output_schema(node: Node) -> type[BaseModel] | None:
    """The pydantic model for the schema a node's tests name, or None for free text."""
    refs = [test["schema"] for test in node.tests if "schema" in test]
    return load_schema(refs[0]) if refs else None


@lru_cache(maxsize=None)
def load_schema(ref: str) -> type[BaseModel]:
    """Load prompts/<ref>, a JSON Schema object, as a pydantic model."""
    schema = json.loads((PROMPTS / ref).read_text())
    return model_from_json_schema(schema, schema.get("title") or Path(ref).stem.title())


def model_from_json_schema(schema: Mapping[str, Any], name: str) -> type[BaseModel]:
    """A pydantic model for the JSON Schema subset prompts use: objects with typed
    properties, enums, arrays and nested objects. additionalProperties false forbids extras."""
    if schema.get("type") != "object":
        raise ValueError(f"{name}: a schema must describe an object")
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for prop, spec in schema.get("properties", {}).items():
        annotation = _annotation(spec, f"{name}_{prop}")
        description = spec.get("description")
        if prop in required:
            fields[prop] = (annotation, Field(description=description))
        else:
            fields[prop] = (Optional[annotation], Field(default=None, description=description))
    extra = "forbid" if schema.get("additionalProperties") is False else "ignore"
    return create_model(name, __config__=ConfigDict(extra=extra), **fields)


def _annotation(spec: Mapping[str, Any], name: str) -> Any:
    """The Python type for one JSON Schema property."""
    if "enum" in spec:
        return Literal[tuple(spec["enum"])]
    kind = spec.get("type")
    if kind in _SCALARS:
        return _SCALARS[kind]
    if kind == "array":
        return list[_annotation(spec.get("items", {}), name)]
    if kind == "object":
        return model_from_json_schema(spec, name) if "properties" in spec else dict[str, Any]
    if kind is None:
        return Any
    raise ValueError(f"{name}: unsupported JSON Schema type {kind!r}")


class MockLLM:
    """Deterministic stand-in for a chat model: one function per prompt template, latency
    from config/mock.yaml times scale, and tokens estimated as characters divided by 4."""

    def __init__(self, tasks: Mapping[str, Callable[[dict[str, Any]], Any]],
                 latency_ms: Mapping[str, float], model_names: Mapping[str, str],
                 scale: float = 1.0) -> None:
        self._tasks = tasks
        self._latency_ms = latency_ms
        self._model_names = model_names
        self._scale = scale

    async def invoke(self, node: Node, prompt: str,
                     schema: type[BaseModel] | None) -> tuple[dict[str, Any], Usage]:
        """Answer with the task function for the node's prompt, after its tier's latency."""
        task = self._tasks.get(Path(str(node.prompt)).stem)
        if task is None:
            raise LookupError(f"no mock task for prompt {node.prompt!r}")
        result = task(extract_inputs(prompt))
        if Path(str(node.prompt)).stem == "classify_segment" and node.model == "small":
            if extract_inputs(prompt).get("applicant.employment") == "contract":
                result = {"label": "unclear"}  # schema rejection exercises the cascade
        # Deliberately unsafe demo fixture: a weak risk model mishandles the refer band.
        if Path(str(node.prompt)).stem == "risk_summary" and node.model == "small":
            result = dict(result)
            if result.get("decision") == "refer":
                result["decision"] = "approve"
                result["summary"] = "Approve: weak-model refer-band error."
        await asyncio.sleep(self._latency_ms[str(node.model)] * self._scale / 1000)
        if schema is not None:
            try:
                output = schema.model_validate(result).model_dump(mode="json")
            except Exception as exc:
                usage = Usage(tokens_in=estimate_tokens(prompt),
                              tokens_out=estimate_tokens(json.dumps(result)),
                              model=self._model_names[str(node.model)], calls=1)
                raise InvalidStructuredOutput(str(exc), usage) from exc
        else:
            output = {"text": str(result)}
        usage = Usage(tokens_in=estimate_tokens(prompt),
                      tokens_out=estimate_tokens(json.dumps(output)),
                      model=self._model_names[str(node.model)], calls=1)
        return output, usage


# Environment variables that hold each live provider's API key, preferred first.
API_KEY_VARS = {"gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"), "groq": ("GROQ_API_KEY",)}
_RETRY_HINT = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)


def is_retryable(exc: BaseException) -> bool:
    """Rate limits and transient server errors are worth retrying; nothing else is."""
    from langchain_core.exceptions import ModelRateLimitError

    if isinstance(exc, ModelRateLimitError):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    return any(marker in str(exc) for marker in ("RESOURCE_EXHAUSTED", "UNAVAILABLE"))


def is_daily_quota(exc: BaseException) -> bool:
    """A daily cap cannot be solved by retrying during this run."""
    message = str(exc).lower()
    return any(marker in message for marker in (
        "perday", "per_day", "requests per day", "per day limit", "daily quota", "tokens per day"))


def retry_delay_s(exc: BaseException, attempt: int) -> float:
    """Wait before retry number `attempt`: the provider's 'retry in Ns' hint when it gives
    one, else exponential backoff capped at a minute."""
    hint = _RETRY_HINT.search(str(exc))
    if hint:
        return min(float(hint.group(1)) + 1.0, 90.0)
    return min(2.0 ** attempt, 60.0)


class LangChainLLM:
    """Live chat models through LangChain at temperature 0. A semaphore caps concurrent calls
    to the provider; rate-limit and transient errors are retried with backoff and counted in
    Usage.retries. Structured nodes use with_structured_output(include_raw=True), so token
    usage comes from the raw reply."""

    def __init__(self, provider: str, model_names: Mapping[str, str],
                 build: Callable[[str, str], Any], max_concurrency: int = 4,
                 max_retries: int = 4,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 quota: QuotaGuard | None = None) -> None:
        self.provider = provider
        self._model_names = model_names
        self._build = build
        self._chats: dict[str, Any] = {}
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._sleep = sleep
        self._quota = quota

    async def invoke(self, node: Node, prompt: str,
                     schema: type[BaseModel] | None) -> tuple[dict[str, Any], Usage]:
        """Measure provider work and waits separately; account for failed replies too."""
        tier = str(node.model)
        totals = Usage(model=self._model_names[tier])
        quota_prompt = prompt + (json.dumps(schema.model_json_schema()) if schema else "")

        def finish(usage: Usage) -> Usage:
            for key in ("calls", "quota_wait_ms", "retry_wait_ms", "provider_ms", "retries"):
                setattr(usage, key, getattr(totals, key))
            return usage

        try:
            async with self._semaphore:
                budget_deadline = time.monotonic() + 300
                for attempt in range(self._max_retries + 1):
                    totals.retries = attempt
                    while True:
                        try:
                            ticket = (self._quota.reserve(
                                self.provider, self._model_names[tier], quota_prompt,
                                output_tokens=node.max_output_tokens) if self._quota else None)
                            break
                        except QuotaExceeded as exc:
                            if (exc.retry_after_s is None or
                                    time.monotonic() + exc.retry_after_s > budget_deadline):
                                raise LiveProviderError(str(exc)) from None
                            totals.quota_wait_ms += await self._pause(node, "quota", exc.retry_after_s)
                    started = time.perf_counter()
                    totals.calls += 1
                    try:
                        result, usage = await self._call(
                            self._chat(tier, node.max_output_tokens), tier, prompt, schema, attempt)
                    except InvalidStructuredOutput as exc:
                        totals.provider_ms += (time.perf_counter() - started) * 1000
                        if ticket is not None:
                            self._quota.settle(ticket, exc.usage.tokens_in + exc.usage.tokens_out)
                        exc.usage = finish(exc.usage)
                        raise
                    except Exception as exc:
                        totals.provider_ms += (time.perf_counter() - started) * 1000
                        if is_daily_quota(exc):
                            if self._quota:
                                self._quota.block(self.provider, self._model_names[tier])
                            raise LiveProviderError(
                                f"{self.provider} daily model quota exhausted; choose another "
                                "provider or Mock") from None
                        if attempt == self._max_retries or not is_retryable(exc):
                            if is_retryable(exc):
                                raise LiveProviderError(
                                    f"{self.provider} model rate limit reached; choose another "
                                    "provider or retry later") from None
                            raise
                        totals.retry_wait_ms += await self._pause(
                            node, "retry", retry_delay_s(exc, attempt + 1))
                    else:
                        totals.provider_ms += (time.perf_counter() - started) * 1000
                        if ticket is not None:
                            self._quota.settle(ticket, usage.tokens_in + usage.tokens_out)
                        return result, finish(usage)
        except Exception as exc:
            if not isinstance(exc, InvalidStructuredOutput):
                totals.usage_complete = totals.calls == 0
                exc.usage = totals
            raise
        raise AssertionError("unreachable")

    async def _pause(self, node: Node, reason: str, seconds: float) -> float:
        """Publish a visible waiting state and return the actual elapsed wait."""
        callback = LLM_EVENTS.get()
        if callback:
            await callback({"type": "node_wait", "node_id": node.id,
                            "reason": reason, "wait_seconds": seconds})
        started = time.perf_counter()
        await self._sleep(seconds)
        elapsed = (time.perf_counter() - started) * 1000
        if callback:
            await callback({"type": "node_resume", "node_id": node.id, "reason": reason})
        return elapsed

    async def _call(self, chat: Any, tier: str, prompt: str, schema: type[BaseModel] | None,
                    retries: int) -> tuple[dict[str, Any], Usage]:
        if schema is None:
            message = await chat.ainvoke(prompt)
            return {"text": message.text}, self._usage(message, tier, retries)
        structured = (chat.with_structured_output(schema, include_raw=True,
                                                 method="json_schema", strict=True)
                      if self.provider == "groq" else
                      chat.with_structured_output(schema, include_raw=True))
        result = await structured.ainvoke(prompt)
        usage = self._usage(result.get("raw"), tier, retries)
        parsed, error = result.get("parsed"), result.get("parsing_error")
        if error is not None or parsed is None:
            raise InvalidStructuredOutput(f"reply did not match {schema.__name__}: {error}", usage)
        try:
            return schema.model_validate(parsed).model_dump(mode="json"), usage
        except ValidationError as exc:
            raise InvalidStructuredOutput(str(exc), usage) from exc

    def _chat(self, tier: str, output_tokens: int = 1024) -> Any:
        key = f"{tier}:{output_tokens}"
        if key not in self._chats:
            chat = self._build(self._model_names[tier], tier)
            if hasattr(chat, "model_copy"):
                field = "max_output_tokens" if self.provider == "gemini" else "max_tokens"
                chat = chat.model_copy(update={field: output_tokens})
            self._chats[key] = chat
        return self._chats[key]

    def _usage(self, message: Any, tier: str, retries: int) -> Usage:
        meta = getattr(message, "usage_metadata", None) or {}
        return Usage(tokens_in=int(meta.get("input_tokens", 0)),
                     tokens_out=int(meta.get("output_tokens", 0)),
                     model=self._model_names[tier], retries=retries, calls=1,
                     usage_complete="input_tokens" in meta and "output_tokens" in meta)


def chat_builder(provider: str, api_key: str,
                 settings: Mapping[str, Any]) -> Callable[[str, str], Any]:
    """Build a provider's LangChain chat model for a model name and tier, always at
    temperature 0, with retries left to LangChainLLM so they are counted."""
    thinking = settings.get("thinking_level") or {}

    def build(model_name: str, tier: str) -> Any:
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            options = {"thinking_level": thinking[tier]} if tier in thinking else {}
            # max_retries counts attempts in the Gemini SDK, so 1 means no hidden retries
            return ChatGoogleGenerativeAI(model=model_name, temperature=0,
                                          google_api_key=api_key, max_retries=1,
                                          max_tokens=1024, timeout=60, **options)
        if provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(model=model_name, temperature=0, api_key=api_key,
                            max_retries=0, max_tokens=1024, timeout=60, reasoning_effort="low")
        raise ValueError(f"unknown live provider {provider!r}")

    return build


def live_llm(provider: str, settings: Mapping[str, Any],
             model: str | None = None, quota_path: str | Path | None = None) -> LangChainLLM:
    """The live adapter for a provider in config/models.yaml, keyed from the environment."""
    if provider not in API_KEY_VARS:
        raise ValueError(f"unknown live provider {provider!r}")
    names = API_KEY_VARS[provider]
    key = next((os.environ[name] for name in names if os.environ.get(name)), None)
    if key is None:
        raise LookupError(f"--live with {provider} needs {names[0]} in .env")
    names = ({"small": model, "large": model} if model else
             {"small": settings["small"], "large": settings["large"]})
    return LangChainLLM(provider, names,
                        chat_builder(provider, key, settings),
                        int(settings.get("max_concurrency", 4)),
                        int(settings.get("max_retries", 4)),
                        quota=QuotaGuard(quota_path, advisory_daily=True) if quota_path
                        else QuotaGuard(advisory_daily=True))
