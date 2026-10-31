"""M10-M11: local demo routes and measured ablation output."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from bench.runner import RESULTS_DIR, benchmark
from mock_services import MOCK_TASKS
from synapse import ROOT
from synapse.api import create_app
from synapse.backends.llm import MockLLM
from synapse.backends.retrieval import KeywordRetriever
from synapse.compiler import compile_workflow
from synapse.config import load_config
from synapse.parser import load_workflow
from synapse.store import Store
from synapse.validator import Check


def test_dashboard_compile_and_websocket_events(tmp_path) -> None:
    app = create_app(db_path=tmp_path / "api.db", results_dir=tmp_path / "results",
                     backend_options={"retriever": KeywordRetriever()})
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200 and "SYNAPSE" in page.text
        assert "sourceNotice" in page.text
        assert client.get("/ui/app.js?v=11").status_code == 200
        assert client.get("/ui/style.css?v=11").status_code == 200
        models = client.get("/models").json()
        assert {m["provider"] for m in models} == {"mock", "gemini", "groq"}
        workflows = client.get("/workflows").json()
        assert {row["name"] for row in workflows} >= {"loan_prescreen", "complaint_triage"}
        compiled = client.post("/compile", json={"workflow": "loan_prescreen"}).json()
        assert len(compiled["source"]["nodes"]) == 11
        assert len(compiled["plan"]["workflow"]["nodes"]) == 9
        assert compiled["graph"]["source"]["nodes"]
        for mode, count in (("baseline", 11), ("compiled", 9)):
            started = client.post("/runs", json={"workflow": "loan_prescreen",
                                                  "mode": mode, "case_id": "A01",
                                                  "scale": 0.001}).json()
            run_id = started["run_id"]
            with client.websocket_connect(f"/ws/runs/{run_id}") as websocket:
                events = []
                while True:
                    event = websocket.receive_json()
                    events.append(event)
                    if event["type"] == "run_end":
                        break
            assert sum(e["type"] == "node_start" for e in events) == count
            assert sum(e["type"] == "node_end" for e in events) == count
            assert events[-1]["run_id"] == run_id and events[-1]["error"] is None
            record = client.get(f"/runs/{run_id}").json()
            assert record["status"] == "done" and len(record["nodes"]) == count
            assert record["key_value"] == "approve"
            totals = record["usage_totals"]
            assert totals["tokens_total"] == totals["tokens_in"] + totals["tokens_out"]
            assert totals["cost"] == pytest.approx(sum(n["usage"]["cost"]
                                                       for n in record["nodes"]))


def test_repeated_case_reuses_model_results_but_writes_once_per_run(tmp_path) -> None:
    from mock_services.bank_api import create_app as bank_app
    bank = bank_app(scale=.001)
    app = create_app(db_path=tmp_path / "api.db", results_dir=tmp_path / "results",
                     backend_options={"retriever": KeywordRetriever(), "bank_app": bank})
    with TestClient(app) as client:
        client.post("/compile", json={"workflow": "complaint_triage"})
        bank.state.ticket_log.clear()  # exclude measured profiling cases
        records = []
        for cache_mode in ("warm", "warm", "cold"):
            response = client.post("/runs", json={"workflow": "complaint_triage",
                                  "mode": "compiled", "case_id": "C01", "scale": .001,
                                  "cache_mode": cache_mode})
            assert response.status_code == 200
            id = response.json()["run_id"]
            with client.websocket_connect(f"/ws/runs/{id}") as websocket:
                while websocket.receive_json()["type"] != "run_end":
                    pass
            records.append(client.get(f"/runs/{id}").json())
            assert records[-1]["status"] == "done"
        assert len(bank.state.ticket_log) == 3
        assert [r["cold_cache"] for r in records] == [True, False, True]
        assert records[0]["usage_totals"]["tokens_total"] > 0
        assert records[1]["usage_totals"]["tokens_total"] == 0
        assert records[2]["usage_totals"]["tokens_total"] > 0
        assert records[0]["key_value"] == records[1]["key_value"] == records[2]["key_value"]
        assert all(next(n for n in r["nodes"] if n["node_id"] == "log_ticket")["cache_hit"] is False
                   for r in records)
        with Store(tmp_path / "api.db") as store:
            restored = store.load_run(records[0]["run_id"])
            assert restored.nodes[0].usage.model_dump() == records[0]["nodes"][0]["usage"]


def test_bundled_benchmark_example_covers_every_workflow() -> None:
    data = json.loads((ROOT / "bench" / "sample_mock.json").read_text())
    assert {row["workflow"] for row in data} == {
        "loan_prescreen", "complaint_triage", "research_agent", "research_llm"}
    assert all(row["sample"] and row["mode"] == "mock" for row in data)
    for result in data:
        cold = [row for row in result["configurations"] if row["cache"] == "cold"]
        assert cold[0]["name"] == "baseline" and cold[-1]["name"] == "memoise"
        assert cold[-1]["correct"] == cold[-1]["runs"]


def test_dashboard_uses_bundled_mock_example_when_no_local_benchmark(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("synapse.api.results_path",
                        lambda _live, _directory: tmp_path / "missing.json")
    app = create_app(db_path=tmp_path / "api.db", results_dir=RESULTS_DIR)
    with TestClient(app) as client:
        data = client.get("/bench/latest").json()
        assert len(data) == 4 and all(row["sample"] for row in data)
        assert client.get("/bench/latest?mode=live").status_code == 404


def test_benchmark_has_cold_warm_ablation_and_correctness() -> None:
    wf = load_workflow(ROOT / "workflows" / "loan_prescreen.yaml")
    result = asyncio.run(benchmark(wf, repeats=2, scale=0.001, case_ids=["A01", "A03"]))
    assert [(c["name"], c["cache"]) for c in result["configurations"]] == [
        ("baseline", "cold"), ("baseline", "warm"),
        ("dead_nodes", "cold"), ("cse", "cold"), ("compact_inputs", "cold"),
        ("parallelise", "cold"), ("routing", "cold"),
        ("memoise", "cold"), ("memoise", "warm")]
    assert all(c["runs"] == 4 and c["correct"] == 4 and c["errors"] == 0
               for c in result["configurations"])


def test_requested_plan_failure_does_not_erase_accepted_routing_proof(
        tmp_path, monkeypatch) -> None:
    from synapse import validator

    db = tmp_path / "api.db"
    app = create_app(db_path=db, results_dir=tmp_path / "results",
                     backend_options={"retriever": KeywordRetriever()})
    lab = app.state.lab
    name, target = lab.target("complaint_triage")
    accepted = compile_workflow(target.workflow,
                                passes=("dead_nodes", "cse", "compact_inputs", "parallelise", "memoise", "routing"),
                                routing_probe=True)
    requested = compile_workflow(target.workflow, passes=())
    lab.remember(name, requested)

    async def approved(*_args, **_kwargs):
        return accepted, [Check(case_id=case["case_id"], node_id="triage.category",
                                kind="decision", passed=True) for case in target.cases], 0.0

    monkeypatch.setattr(validator, "validate_workflow", approved)
    monkeypatch.setattr(validator, "validate", lambda *_args, **_kwargs: [
        Check(case_id="C01", node_id="triage.category", kind="decision", passed=False)])
    with TestClient(app) as client:
        response = client.post("/validate", json={"workflow": name,
                                                  "plan_id": requested.plan_id, "scale": 0})
    assert response.status_code == 200
    assert response.json()["passed"] == 0
    with Store(db) as store:
        assert store.route_validated(target.workflow)


def test_live_run_needs_proof_for_its_actual_provider_and_models(tmp_path) -> None:
    db = tmp_path / "api.db"
    lab = create_app(db_path=db, results_dir=tmp_path / "results").state.lab
    _, target = lab.target("loan_prescreen")
    routed = compile_workflow(target.workflow,
                              passes=("dead_nodes", "cse", "compact_inputs", "parallelise", "memoise", "routing"),
                              routing_probe=True)
    assert any(node.fallback_model for node in routed.workflow.nodes)
    safe = lab.live_plan(target, routed, explicit=False)
    assert not any(node.fallback_model for node in safe.workflow.nodes)
    with pytest.raises(HTTPException, match="has not passed validation"):
        lab.live_plan(target, routed, explicit=True)

    models = load_config("models.yaml")
    provider = models["live_provider"]
    tiers = models["providers"][provider]
    names = {tier: str(tiers[tier]) for tier in ("small", "large")}
    with Store(db) as store:
        store.save_validation(target.workflow, routed,
                              [case["case_id"] for case in target.cases], True,
                              provider=provider, model_names=names)
    assert lab.live_plan(target, routed, explicit=True) is routed


def test_dashboard_can_choose_groq_without_real_network(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("synapse.backends.load_env", lambda: None)
    fake = MockLLM(MOCK_TASKS, {"small": 0, "large": 0},
                   load_config("models.yaml")["providers"]["groq"], scale=0)
    app = create_app(db_path=tmp_path / "api.db", results_dir=tmp_path / "results",
                     backend_options={"retriever": KeywordRetriever(), "llm": fake})
    with TestClient(app) as client:
        invalid = client.post("/runs", json={"workflow": "complaint_triage", "case_id": "C01",
                                               "provider": "mock", "live": True})
        assert invalid.status_code == 422
        started = client.post("/runs", json={"workflow": "complaint_triage", "case_id": "C01",
                                               "provider": "groq", "model": "openai/gpt-oss-20b",
                                               "scale": 0}).json()
        assert started["provider"] == "groq" and started["live"]
        assert started["model"] == "openai/gpt-oss-20b"
        with client.websocket_connect(f"/ws/runs/{started['run_id']}") as websocket:
            while websocket.receive_json()["type"] != "run_end":
                pass
        record = client.get(f"/runs/{started['run_id']}").json()
        assert record["status"] == "done" and record["key_value"] == "failed_transfer"
        assert record["provider"] == "groq"
