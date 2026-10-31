"""M3: measured node profiles and lossless SQLite run traces."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from mock_services import suite_for
from synapse import ROOT
from synapse.backends import mock_backends
from synapse.backends.retrieval import KeywordRetriever
from synapse.cost import estimate_cost
from synapse.config import load_config
from synapse.executor import execute
from synapse.ir import Usage
from synapse.parser import load_workflow
from synapse.profiler import build_profile
from synapse.store import Store


LOAN = load_workflow(ROOT / "workflows" / "loan_prescreen.yaml")


def test_cost_uses_configured_mock_units_and_live_prices() -> None:
    usage = Usage(tokens_in=100, tokens_out=50, model="mock-large")
    assert estimate_cost(usage, "large") == 1.5
    assert estimate_cost(usage, "small") == 0.15
    models = load_config("models.yaml")["providers"]
    for provider, expected_small, expected_large in (
            ("gemini", 0.0001, 0.0002625),
            ("groq", 0.0000225, 0.000045)):
        usage.model = models[provider]["small"]
        assert estimate_cost(usage, "large", provider=provider) == pytest.approx(expected_small)
        usage.model = models[provider]["large"]
        assert estimate_cost(usage, "small", provider=provider) == pytest.approx(expected_large)
    usage.model = "unknown-live-model"
    with pytest.raises(ValueError, match="not configured"):
        estimate_cost(usage, "large", provider="groq")


def test_profile_and_store_repeat_runs(tmp_path) -> None:
    case = suite_for("loan_prescreen").case("A01")

    async def collect():
        async with mock_backends(0.001, retriever=KeywordRetriever()) as backends:
            return [await execute(LOAN, case, "baseline", backends) for _ in range(2)]

    records = asyncio.run(collect())
    profile = build_profile(LOAN, records)
    assert profile.runs == 2
    assert profile.critical_path[0] in {"parse_app", "fetch_statement"}
    assert profile.critical_path[-1] == "notify_customer"
    assert profile.nodes["bureau_check"].repeat_input_rate == 0.5
    assert profile.nodes["risk_summary"].cost > 0
    assert profile.nodes["risk_summary"].tokens_in > 0
    assert all(n.input_hash and len(n.input_hash) == 64 for r in records for n in r.nodes)
    assert all(n.cpu_pct is not None and n.mem_mb is not None for r in records for n in r.nodes)

    with Store(tmp_path / "results.db") as store:
        for record in records:
            store.save_run(record)
        assert store.list_runs("loan_prescreen") == [r.run_id for r in records]
        restored = store.load_run(records[0].run_id)
        assert restored == records[0]
        with pytest.raises(sqlite3.IntegrityError):
            store.save_run(records[0])
        assert len(store.load_run(records[0].run_id).nodes) == len(LOAN.nodes)


def test_profile_requires_runs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_profile(LOAN, [])
