"""The second banking workflow runs without loan-specific compiler rules."""

from __future__ import annotations

import asyncio

from mock_services import complaints, suite_for
from mock_services.bank_api import create_app
from synapse import ROOT
from synapse.backends import mock_backends
from synapse.backends.retrieval import KeywordRetriever
from synapse.compiler import compile_workflow
from synapse.executor import execute, final_outputs, resolve
from synapse.graph import build_graph
from synapse.ir import Effect, NodeKind
from synapse.parser import load_workflow


WORKFLOW = load_workflow(ROOT / "workflows" / "complaint_triage.yaml")
SUITE = suite_for(WORKFLOW.name)


def test_complaint_workflow_has_valid_dependencies_and_safe_effects() -> None:
    graph = build_graph(WORKFLOW)
    assert len(graph) == 5
    assert WORKFLOW.outputs == ["triage", "reply", "ticket"]
    assert [(node.id, node.kind, node.effect) for node in WORKFLOW.nodes] == [
        ("classify_complaint", NodeKind.LLM, Effect.PURE),
        ("fetch_transactions", NodeKind.TOOL, Effect.READ),
        ("policy_lookup", NodeKind.RETRIEVE, Effect.PURE),
        ("draft_reply", NodeKind.LLM, Effect.PURE),
        ("log_ticket", NodeKind.TOOL, Effect.WRITE),
    ]
    assert set(graph.predecessors("draft_reply")) == {
        "classify_complaint", "fetch_transactions", "policy_lookup"}
    assert set(graph.predecessors("log_ticket")) == {"classify_complaint", "draft_reply"}


def test_all_complaints_match_expected_route_and_log_one_ticket_per_run() -> None:
    rows = complaints.load_complaints()
    assert [row["case_id"] for row in rows] == [f"C{n:02d}" for n in range(1, 18)]
    assert len({row["expected_category"] for row in rows}) >= 5
    plan = compile_workflow(WORKFLOW)
    assert any(d.pass_name == "parallelise" and d.status == "applied"
               for d in plan.decisions)
    assert next(node for node in plan.workflow.nodes if node.id == "log_ticket") == \
        next(node for node in WORKFLOW.nodes if node.id == "log_ticket")
    assert "log_ticket" not in plan.cached_nodes

    async def collect() -> None:
        bank = create_app(scale=0.001)
        async with mock_backends(0.001, bank_app=bank,
                                 retriever=KeywordRetriever()) as backends:
            for row in rows:
                case = SUITE.case(row["case_id"])
                baseline = await execute(WORKFLOW, case, "baseline", backends)
                compiled = await execute(WORKFLOW, case, "compiled", backends, plan=plan)
                for run in (baseline, compiled):
                    assert not [node.error for node in run.nodes if node.error], row["case_id"]
                    assert sum(node.node_id == "log_ticket" for node in run.nodes) == 1
                    result = final_outputs(run)
                    assert resolve(SUITE.key_field, result) == SUITE.expected(row["case_id"])
                    assert result["triage"]["team"] == row["expected_team"]
                    assert result["reply"]["transaction_ids"] == row["expected_transactions"]
                    assert result["ticket"]["team"] == row["expected_team"]
                assert final_outputs(baseline) == final_outputs(compiled)
                assert len(bank.state.ticket_log) == 2 * int(row["case_id"][1:])

    asyncio.run(collect())
