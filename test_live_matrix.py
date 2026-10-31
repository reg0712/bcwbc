"""Every workflow and selectable live model through the API, with no network calls."""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from mock_services import MOCK_TASKS
from synapse.api import create_app
from synapse.backends.llm import LangChainLLM, estimate_tokens, extract_inputs
from synapse.backends.retrieval import KeywordRetriever
from synapse.config import load_config
from synapse import ROOT
from synapse.backends import live_backends
from synapse.compiler import compile_workflow
from synapse.executor import execute, final_outputs
from bench.runner import load_target


TASK_FOR_SCHEMA = {
    "Applicant": "parse_app",
    "Cashflow": "statement_analysis",
    "Segment": "classify_segment",
    "RiskAssessment": "risk_summary",
    "Triage": "classify_complaint",
    "ComplaintReply": "draft_reply",
    "ResearchAnswer": "research_answer",
}


class FakeStructuredChat:
    """Exercise the LangChain adapter using the real prompt and output schemas."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.schema: Any = None

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeStructuredChat:
        assert kwargs["include_raw"] is True
        assert (kwargs.get("method") == "json_schema") == (self.provider == "groq")
        if self.provider == "groq":
            assert kwargs["strict"] is True
        reply = FakeStructuredChat(self.provider)
        reply.schema = schema
        return reply

    async def ainvoke(self, prompt: str) -> dict[str, Any]:
        task = MOCK_TASKS[TASK_FOR_SCHEMA[self.schema.__name__]]
        value = self.schema.model_validate(task(extract_inputs(prompt)))
        output = json.dumps(value.model_dump(mode="json"))
        raw = AIMessage(content=output, usage_metadata={
            "input_tokens": estimate_tokens(prompt),
            "output_tokens": estimate_tokens(output),
            "total_tokens": estimate_tokens(prompt) + estimate_tokens(output)})
        return {"raw": raw, "parsed": value, "parsing_error": None}


class FailingChat(FakeStructuredChat):
    async def ainvoke(self, prompt: str) -> dict[str, Any]:
        raise ValueError("synthetic model failure")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FailingChat:
        return self


@pytest.mark.parametrize("provider,tier", [
    ("gemini", "small"), ("gemini", "large"), ("groq", "small"), ("groq", "large")])
def test_every_case_through_each_model_adapter(provider, tier, monkeypatch) -> None:
    monkeypatch.setattr("synapse.backends.load_env", lambda: None)
    model = load_config("models.yaml")["providers"][provider][tier]

    async def check():
        for path in ("loan_prescreen.yaml", "complaint_triage.yaml", "research_llm_langgraph.py"):
            target = load_target(ROOT / "workflows" / path)
            wf = target.workflow
            adapter = LangChainLLM(provider, {"small": model, "large": model},
                                   lambda *_: FakeStructuredChat(provider))
            async with live_backends(.001, provider=provider, model=model, llm=adapter,
                                     retriever=KeywordRetriever(), functions=target.functions) as backends:
                for case in target.cases:
                    before = await execute(wf, case, "baseline", backends, live=True)
                    after = await execute(wf, case, "compiled", backends,
                                          plan=compile_workflow(wf), live=True)
                    assert not any(n.error for r in (before, after) for n in r.nodes)
                    old, new = final_outputs(before), final_outputs(after)
                    assert {k: old[k] for k in wf.outputs} == {k: new[k] for k in wf.outputs}
                    assert all(n.usage.model == model for n in after.nodes if n.usage.calls)
    asyncio.run(check())


@pytest.mark.parametrize("provider,tier", [
    ("gemini", "small"), ("gemini", "large"),
    ("groq", "small"), ("groq", "large")])
def test_all_workflows_with_each_selected_model(
        provider: str, tier: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("synapse.backends.load_env", lambda: None)
    model = load_config("models.yaml")["providers"][provider][tier]
    adapter = LangChainLLM(provider, {"small": model, "large": model},
                           lambda _name, _tier: FakeStructuredChat(provider))
    app = create_app(db_path=tmp_path / "api.db", results_dir=tmp_path / "results",
                     backend_options={"llm": adapter, "retriever": KeywordRetriever()})
    with TestClient(app) as client:
        for workflow in ("loan_prescreen", "complaint_triage", "research_agent", "research_llm"):
            case_id = next(row["cases"][0] for row in client.get("/workflows").json()
                           if row["name"] == workflow)
            records = []
            for mode in ("baseline", "compiled"):
                started = client.post("/runs", json={"workflow": workflow, "mode": mode,
                                  "case_id": case_id, "provider": provider,
                                  "model": model, "scale": 0})
                assert started.status_code == 200, started.text
                run_id = started.json()["run_id"]
                with client.websocket_connect(f"/ws/runs/{run_id}") as websocket:
                    while websocket.receive_json()["type"] != "run_end":
                        pass
                record = client.get(f"/runs/{run_id}").json()
                assert record["status"] == "done", record.get("error")
                assert not any(node["error"] for node in record["nodes"])
                assert record["usage_totals"]["tokens_total"] == sum(
                    node["usage"]["tokens_in"] + node["usage"]["tokens_out"]
                    for node in record["nodes"])
                assert record["usage_totals"]["cost"] == pytest.approx(sum(
                    node["usage"]["cost"] for node in record["nodes"]))
                records.append(record)
            if workflow == "research_agent":
                assert all(row["provider"] == "mock" and row["model"] is None
                           for row in records)
                assert records[0]["usage_totals"]["tokens_total"] == 0
                assert records[0]["nodes"][-1]["output"] == records[1]["nodes"][-1]["output"]
            else:
                assert records[0].get("key_value") == records[1].get("key_value")
                assert records[0]["usage_totals"]["tokens_total"] > 0
                rates = load_config("prices.yaml")["live_usd_per_1k_tokens"][provider][tier]
                for record in records:
                    usage = record["usage_totals"]
                    assert usage["cost"] == pytest.approx(
                        (usage["tokens_in"] * rates["in"] +
                         usage["tokens_out"] * rates["out"]) / 1000)


def test_failed_model_call_has_failed_status(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("synapse.backends.load_env", lambda: None)
    model = load_config("models.yaml")["providers"]["groq"]["small"]
    adapter = LangChainLLM("groq", {"small": model, "large": model},
                           lambda _name, _tier: FailingChat("groq"))
    app = create_app(db_path=tmp_path / "api.db", results_dir=tmp_path / "results",
                     backend_options={"llm": adapter, "retriever": KeywordRetriever()})
    with TestClient(app) as client:
        started = client.post("/runs", json={"workflow": "complaint_triage",
                               "case_id": "C01", "provider": "groq", "model": model,
                               "scale": 0}).json()
        with client.websocket_connect(f"/ws/runs/{started['run_id']}") as websocket:
            while websocket.receive_json()["type"] != "run_end":
                pass
        record = client.get(f"/runs/{started['run_id']}").json()
        assert record["status"] == "failed"
        assert "synthetic model failure" in record["error"]
