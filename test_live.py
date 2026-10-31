"""Live LLM path with fake chat models (never the network): structured replies and token
usage, counted rate-limit retries, schema failures, Gemini settings, API keys from .env, and
live calls priced in US dollars."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import AIMessage

import synapse.backends as backends_module
from synapse import ROOT
from synapse.backends import live_backends, open_backends
from synapse.backends.llm import (InvalidStructuredOutput, LangChainLLM, LiveProviderError,
                                  MockLLM, chat_builder,
                                  load_schema, output_schema, render_prompt, LLM_EVENTS, is_daily_quota)
from synapse.backends.quota import QuotaExceeded
from synapse.backends.retrieval import KeywordRetriever
from synapse.config import load_config, load_env
from synapse.executor import execute
from synapse.ir import Workflow
from synapse.parser import load_workflow
from mock_services import MOCK_TASKS

LOAN = load_workflow(ROOT / "workflows" / "loan_prescreen.yaml")
CLASSIFY = next(node for node in LOAN.nodes if node.id == "classify_segment")
SEGMENT_INPUTS = {"applicant.income": 140000, "applicant.employment": "salaried"}
NAMES = {"small": "gemini-small-test", "large": "gemini-large-test"}


class FakeChat:
    """A stand-in for a LangChain chat model that answers from a script of replies; an
    exception in the script is raised instead."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def with_structured_output(self, schema: Any, include_raw: bool = False,
                               **kwargs: Any) -> FakeChat:
        assert include_raw, "usage is read from the raw reply"
        return self

    async def ainvoke(self, prompt: str) -> Any:
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def structured(label: str, tokens_in: int = 120, tokens_out: int = 7) -> dict[str, Any]:
    """A with_structured_output(include_raw=True) result for the segment schema."""
    raw = AIMessage(content="{}", usage_metadata={"input_tokens": tokens_in,
                                                  "output_tokens": tokens_out,
                                                  "total_tokens": tokens_in + tokens_out})
    return {"raw": raw, "parsed": load_schema("segment.json")(label=label), "parsing_error": None}


def adapter(chat: FakeChat, max_retries: int = 4) -> tuple[LangChainLLM, list[float]]:
    """A live adapter over a fake chat model, recording backoff waits instead of sleeping."""
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    return LangChainLLM("gemini", NAMES, lambda name, tier: chat, max_retries=max_retries,
                        sleep=record), waits


def classify(llm: LangChainLLM) -> tuple[dict[str, Any], Any]:
    prompt = render_prompt(CLASSIFY, SEGMENT_INPUTS)
    return asyncio.run(llm.invoke(CLASSIFY, prompt, output_schema(CLASSIFY)))


def test_structured_reply_is_parsed_with_the_raw_token_usage() -> None:
    llm, waits = adapter(FakeChat([structured("salaried_prime")]))
    output, usage = classify(llm)
    assert output == {"label": "salaried_prime"}
    assert (usage.tokens_in, usage.tokens_out, usage.model, usage.retries) == (
        120, 7, "gemini-large-test", 0)
    assert waits == []


def test_rate_limits_are_retried_with_the_providers_hint_and_counted() -> None:
    chat = FakeChat([ModelRateLimitError("429 RESOURCE_EXHAUSTED. Please retry in 3s."),
                     structured("salaried_prime")])
    llm, waits = adapter(chat)
    output, usage = classify(llm)
    assert output == {"label": "salaried_prime"}
    assert usage.retries == 1
    assert waits == [4.0]  # the hinted 3 s plus a second of margin


def test_local_minute_budget_waits_without_an_extra_provider_call() -> None:
    chat = FakeChat([structured("salaried_prime")])
    llm, waits = adapter(chat)

    class OnePause:
        calls = 0

        def reserve(self, _provider: str, _model: str, _prompt: str, **kwargs: Any) -> int:
            self.calls += 1
            if self.calls == 1:
                raise QuotaExceeded("local minute budget reached", retry_after_s=0.01)
            return 1

        def settle(self, _ticket: int, _tokens: int) -> None:
            pass

    llm._quota = OnePause()
    output, usage = classify(llm)
    assert output["label"] == "salaried_prime"
    assert usage.retries == 0 and chat.calls == 1
    assert waits == [0.01]


def test_wait_events_measure_elapsed_time_and_failed_schema_settles_usage() -> None:
    events, reservations, settled = [], [], []
    bad = {"raw": AIMessage(content="?", usage_metadata={"input_tokens": 90,
           "output_tokens": 4, "total_tokens": 94}), "parsed": None,
           "parsing_error": ValueError("bad schema")}
    llm = LangChainLLM("gemini", NAMES, lambda *_: FakeChat([bad]))

    class Guard:
        def reserve(self, provider, model, prompt, **kwargs):
            reservations.append((prompt, kwargs))
            if len(reservations) == 1:
                raise QuotaExceeded("wait", .01)
            return 1

        def settle(self, ticket, tokens):
            settled.append(tokens)

    async def event(value):
        events.append(value)

    llm._quota = Guard()
    token = LLM_EVENTS.set(event)
    try:
        with pytest.raises(InvalidStructuredOutput) as caught:
            classify(llm)
    finally:
        LLM_EVENTS.reset(token)
    usage = caught.value.usage
    assert usage.quota_wait_ms >= 9
    assert usage.provider_ms > 0 and usage.calls == 1
    assert [e["type"] for e in events] == ["node_wait", "node_resume"]
    assert settled == [94]
    assert '"properties"' in reservations[0][0]
    assert reservations[0][1]["output_tokens"] == CLASSIFY.max_output_tokens


def test_minute_quota_is_not_mistaken_for_daily_exhaustion() -> None:
    assert not is_daily_quota(RuntimeError("free_tier_requests RequestsPerMinute 429"))
    assert is_daily_quota(RuntimeError("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))


def test_retries_stop_at_the_configured_limit() -> None:
    llm, waits = adapter(FakeChat([ModelRateLimitError("429")] * 3), max_retries=2)
    with pytest.raises(LiveProviderError, match="rate limit reached"):
        classify(llm)
    assert waits == [2.0, 4.0]  # exponential backoff without a hint


def test_daily_quota_fails_immediately_with_short_message() -> None:
    error = ModelRateLimitError("429 free_tier_requests: requests per day limit exceeded")
    llm, waits = adapter(FakeChat([error]), max_retries=4)
    with pytest.raises(LiveProviderError, match="daily model quota exhausted") as caught:
        classify(llm)
    assert "free_tier_requests" not in str(caught.value)
    assert waits == []


def test_other_errors_fail_at_once() -> None:
    chat = FakeChat([ValueError("400 INVALID_ARGUMENT: bad request")])
    llm, waits = adapter(chat)
    with pytest.raises(ValueError, match="INVALID_ARGUMENT"):
        classify(llm)
    assert (chat.calls, waits) == (1, [])


def test_a_reply_that_breaks_the_schema_keeps_its_usage_for_the_fallback() -> None:
    reply = {"raw": AIMessage(content="?", usage_metadata={"input_tokens": 90, "output_tokens": 4,
                                                           "total_tokens": 94}),
             "parsed": None, "parsing_error": ValueError("label 'unclear' is not allowed")}
    llm, _ = adapter(FakeChat([reply]))
    with pytest.raises(InvalidStructuredOutput) as caught:
        classify(llm)
    assert (caught.value.usage.tokens_in, caught.value.usage.tokens_out) == (90, 4)


def test_both_failed_model_tiers_keep_billed_usage(monkeypatch) -> None:
    monkeypatch.setattr(backends_module, "load_env", lambda: None)
    names = load_config("models.yaml")["providers"]["groq"]
    bad = {"raw": AIMessage(content="?", usage_metadata={"input_tokens": 90, "output_tokens": 4,
           "total_tokens": 94}), "parsed": None, "parsing_error": ValueError("invalid")}
    llm = LangChainLLM("groq", names, lambda *_: FakeChat([bad]))
    node = CLASSIFY.model_copy(update={"model": "small", "fallback_model": "large"})
    wf = Workflow(name="fallback_test", inputs=["applicant"], nodes=[node])

    async def check():
        async with live_backends(.001, provider="groq", llm=llm,
                                 retriever=KeywordRetriever()) as backends:
            return await execute(wf, {"case_id": "T1", "applicant": {
                "income": 140000, "employment": "salaried"}}, "baseline", backends, live=True)
    record = asyncio.run(check())
    usage = record.nodes[0].usage
    assert "human review required" in record.nodes[0].error
    assert (usage.tokens_in, usage.tokens_out, usage.calls) == (180, 8, 2)
    assert usage.cost > 0 and usage.usage_complete


def test_gemini_models_run_at_temperature_zero_with_low_thinking_on_the_large_tier() -> None:
    settings = load_config("models.yaml")["providers"]["gemini"]
    build = chat_builder("gemini", "test-key-not-real", settings)
    large, small = build(settings["large"], "large"), build(settings["small"], "small")
    assert (large.temperature, large.thinking_level, large.max_retries) == (0.0, "low", 1)
    assert (small.temperature, small.thinking_level) == (0.0, None)


def test_live_backends_need_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends_module, "load_env", lambda: None)  # never read the real .env
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    async def open_live() -> None:
        async with open_backends(live=True, scale=0.001, retriever=KeywordRetriever()):
            pass
    with pytest.raises(LookupError, match="GOOGLE_API_KEY"):
        asyncio.run(open_live())


def test_live_runs_are_priced_in_dollars_at_the_configured_rates(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends_module, "load_env", lambda: None)
    wf = Workflow(name="toy", inputs=["applicant"], nodes=[CLASSIFY.model_copy(
        update={"inputs": ["applicant.income", "applicant.employment"]})])
    case = {"case_id": "T1", "applicant": {"income": 140000, "employment": "salaried"}}
    names = load_config("models.yaml")["providers"]["gemini"]
    stand_in = MockLLM(MOCK_TASKS, {"small": 300, "large": 1500}, names, scale=0.001)

    async def run_live() -> Any:
        async with live_backends(0.001, llm=stand_in, retriever=KeywordRetriever()) as backends:
            assert (backends.provider, backends.max_concurrency) == ("gemini", 4)
            return await execute(wf, case, "baseline", backends, live=True)
    record = asyncio.run(run_live())
    usage = record.nodes[0].usage
    rates = load_config("prices.yaml")["live_usd_per_1k_tokens"]["gemini"]["large"]
    assert record.live
    assert usage.cost == pytest.approx(
        (usage.tokens_in * rates["in"] + usage.tokens_out * rates["out"]) / 1000)


def test_load_env_never_overrides_the_environment_or_sets_empty_values(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("SYNAPSE_TEST_A=from-file\nSYNAPSE_TEST_B=from-file\nSYNAPSE_TEST_C=\n"
                   "# SYNAPSE_TEST_D=commented\n")
    monkeypatch.setenv("SYNAPSE_TEST_B", "already-set")
    for name in ("SYNAPSE_TEST_A", "SYNAPSE_TEST_C", "SYNAPSE_TEST_D"):
        monkeypatch.delenv(name, raising=False)
    load_env(env)
    assert os.environ["SYNAPSE_TEST_A"] == "from-file"
    assert os.environ["SYNAPSE_TEST_B"] == "already-set"
    assert "SYNAPSE_TEST_C" not in os.environ and "SYNAPSE_TEST_D" not in os.environ
    monkeypatch.delenv("SYNAPSE_TEST_A")
