"""Workflow-aware validation of structured, decision and free-text outputs."""

from __future__ import annotations

import asyncio

from mock_services import suite_for
from synapse import ROOT
from synapse.backends import mock_backends
from synapse.backends.retrieval import KeywordRetriever
from synapse.compiler import compile_workflow
from synapse.executor import execute
from synapse.parser import load_workflow
from synapse.store import Store
from synapse.validator import noise_floor, validate


COMPLAINT = load_workflow(ROOT / "workflows" / "complaint_triage.yaml")


def test_complaint_validation_checks_schema_route_and_bank_reads(monkeypatch) -> None:
    case = suite_for(COMPLAINT.name).case("C01")

    async def original():
        async with mock_backends(0.001, retriever=KeywordRetriever()) as backends:
            return await execute(COMPLAINT, case, "baseline", backends)

    golden = asyncio.run(original())
    changed = golden.model_copy(deep=True)
    changed.nodes[0].output["triage"]["category"] = "other"
    changed.nodes[1].output["history"]["account"] = "TEST-ACC-WRONG"
    checks = validate([golden], [changed], 0, wf=COMPLAINT,
                      key_fields=["triage.category"])
    assert any(c.kind == "decision" and not c.passed for c in checks)
    assert any(c.kind == "exact" and not c.passed and c.node_id == "fetch_transactions"
               for c in checks)

    text_change = golden.model_copy(deep=True)
    text_change.nodes[3].output["reply"]["body"] += " Thank you."
    monkeypatch.setattr("synapse.validator._embedding_model",
                        lambda: (_ for _ in ()).throw(OSError("model unavailable")))
    checks = validate([golden], [text_change], 0, wf=COMPLAINT,
                      key_fields=["triage.category"])
    assert all(c.passed for c in checks)
    assert any("difflib fallback" in c.detail for c in checks)
    assert noise_floor([golden], [text_change]) > 0


def test_routing_proof_requires_entire_current_suite_and_source(tmp_path) -> None:
    cases = suite_for(COMPLAINT.name).cases()
    plan = compile_workflow(COMPLAINT, passes=("dead_nodes", "cse", "compact_inputs", "parallelise",
                                                     "memoise", "routing"), routing_probe=True)
    ids = [case["case_id"] for case in cases]
    with Store(tmp_path / "runs.db") as store:
        store.save_validation(COMPLAINT, plan, ids[:1], True)
        assert not store.route_validated(COMPLAINT)
        store.save_validation(COMPLAINT, plan, ids, False)
        assert not store.route_validated(COMPLAINT)
        store.save_validation(COMPLAINT, plan, ids, True)
        assert store.route_validated(COMPLAINT)
        assert not store.route_validated(COMPLAINT, provider="gemini",
                                         model_names={"small": "small-live", "large": "large-live"})
        changed = COMPLAINT.model_copy(deep=True)
        changed.nodes[0].tests.append({"schema": "changed.json"})
        assert not store.route_validated(changed)

    altered = COMPLAINT.model_copy(deep=True)
    prompt = tmp_path / "classify.txt"
    prompt.write_text("Complaint: {inputs}")
    altered.nodes[0].prompt = str(prompt)
    altered_plan = compile_workflow(altered, passes=("dead_nodes", "cse", "compact_inputs", "parallelise",
                                                     "memoise", "routing"), routing_probe=True)
    with Store(tmp_path / "runs.db") as store:
        store.save_validation(altered, altered_plan, ids, True)
        assert store.route_validated(altered)
        prompt.write_text("Changed complaint: {inputs}")
        assert not store.route_validated(altered)
