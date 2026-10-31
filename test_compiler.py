"""M4-M6: safe rewrites, concurrent execution, cache expiry, and rejection."""

from __future__ import annotations

import asyncio

import pytest

from mock_services import suite_for
from synapse import ROOT
from synapse.backends import mock_backends
from synapse.backends.retrieval import KeywordRetriever
from synapse.compiler import compile_workflow
from synapse.executor import execute
from synapse.ir import Node, Workflow
from synapse.parser import load_workflow
from synapse.passes.memoise import ExactCache, cache_key
from synapse.passes.parallelise import parallel_levels
from synapse.validator import validate_workflow

LOAN = load_workflow(ROOT / "workflows" / "loan_prescreen.yaml")


def test_core_passes_remove_only_safe_nodes_and_preserve_writes() -> None:
    plan = compile_workflow(LOAN)
    ids = [node.id for node in plan.workflow.nodes]
    assert len(ids) == 9
    assert "offers_lookup" not in ids and "policy_lookup_2" not in ids
    assert ids[-1] == "notify_customer"
    assert "notify_customer" not in plan.cached_nodes
    risk = next(n for n in plan.workflow.nodes if n.id == "risk_summary")
    assert "policy_a" in risk.inputs and "policy_b" not in risk.inputs
    assert plan.workflow.outputs == ["risk", "sms"]
    assert any({"kyc_check", "bureau_check"} <= set(group)
               for group in parallel_levels(plan.workflow))


def test_dead_node_pass_skips_workflow_without_declared_outputs() -> None:
    wf = Workflow(name="unknown_final", inputs=["x"], nodes=[
        Node(id="a", kind="transform", effect="pure", inputs=["x"], outputs=["a_out"])])
    plan = compile_workflow(wf, passes=("dead_nodes",))
    assert [n.id for n in plan.workflow.nodes] == ["a"]
    assert plan.decisions[0].status == "skipped"


def test_required_check_survives_and_read_does_not_merge_across_write() -> None:
    wf = Workflow(name="safety", inputs=["x"], outputs=["done"], nodes=[
        Node(id="read_before", kind="tool", tool="balance", effect="read", ttl_s=5,
             deterministic=True, inputs=["x"], outputs=["old"]),
        Node(id="mandatory", kind="tool", tool="audit", effect="read", required=True,
             inputs=["x"], outputs=["audit"]),
        Node(id="write", kind="tool", tool="update", effect="write", inputs=["x"],
             outputs=["done"]),
        Node(id="read_after", kind="tool", tool="balance", effect="read", ttl_s=5,
             deterministic=True, inputs=["x"], outputs=["new"]),
    ])
    plan = compile_workflow(wf, passes=("dead_nodes", "cse"))
    assert "mandatory" in [node.id for node in plan.workflow.nodes]
    # An unconsumed read is removable; require both reads to test the CSE barrier.
    wf.outputs.extend(["old", "new"])
    plan = compile_workflow(wf, passes=("dead_nodes", "cse"))
    assert "read_before" in [n.id for n in plan.workflow.nodes]
    assert "read_after" in [n.id for n in plan.workflow.nodes]


def test_cache_is_exact_expires_and_invalidates_customer(monkeypatch) -> None:
    node = next(n for n in LOAN.nodes if n.id == "bureau_check")
    key = cache_key(node, {"applicant.pan": "P1"})
    assert key != cache_key(node, {"applicant.pan": "P2"})
    clock = [100.0]
    monkeypatch.setattr("synapse.passes.memoise.time.monotonic", lambda: clock[0])
    cache = ExactCache()
    cache.put(key, {"bureau": {"score": 700}}, 10, "C1")
    assert cache.get(key) == {"bureau": {"score": 700}}
    clock[0] = 110.0
    assert cache.get(key) is None
    cache.put(key, {"bureau": {"score": 700}}, 10, "C1")
    cache.invalidate("C2")
    assert cache.get(key) is not None
    cache.invalidate("C1")
    assert cache.get(key) is None


def test_repeat_compiled_run_hits_cache_but_never_caches_sms() -> None:
    case = suite_for("loan_prescreen").case("A01")
    plan = compile_workflow(LOAN)
    cache = ExactCache()

    async def collect():
        async with mock_backends(0.001, retriever=KeywordRetriever()) as backends:
            return [await execute(LOAN, case, "compiled", backends, plan=plan, cache=cache)
                    for _ in range(2)]

    first, second = asyncio.run(collect())
    assert any(n.cache_hit for n in second.nodes)
    assert all(not n.cache_hit for n in first.nodes)
    assert not next(n for n in second.nodes if n.node_id == "notify_customer").cache_hit


def test_ablation_without_parallel_pass_stays_sequential() -> None:
    case = suite_for("loan_prescreen").case("A01")
    plan = compile_workflow(LOAN, passes=("dead_nodes", "cse"))

    async def collect():
        async with mock_backends(0.001, retriever=KeywordRetriever()) as backends:
            return await execute(LOAN, case, "compiled", backends, plan=plan)

    record = asyncio.run(collect())
    assert all(later.start_s >= earlier.end_s for earlier, later in
               zip(record.nodes, record.nodes[1:]))


def test_validator_accepts_safe_passes_and_rejects_unsafe_risk_route() -> None:
    cases = [suite_for("loan_prescreen").case(id) for id in ("A01", "A03")]

    async def check():
        async with mock_backends(0.001, retriever=KeywordRetriever()) as backends:
            return await validate_workflow(LOAN, cases, backends, unsafe_route_risk=True)

    plan, checks, floor = asyncio.run(check())
    assert checks and all(c.passed for c in checks)
    assert floor == 0
    assert any(d.pass_name == "unsafe_route_risk" and d.status == "rejected"
               and "A03" in d.reason for d in plan.decisions)
    assert next(n for n in plan.workflow.nodes if n.id == "risk_summary").model == "large"
    assert next(n for n in plan.workflow.nodes if n.id == "classify_segment").model == "small"


def test_routed_contract_case_escalates_to_large_model() -> None:
    case = next(c for c in suite_for("loan_prescreen").cases()
                if "Employment: contract" in c["application_pdf"])
    plan = compile_workflow(LOAN, passes=("dead_nodes", "cse", "parallelise", "routing"),
                            routing_probe=True)

    async def collect():
        async with mock_backends(0.001, retriever=KeywordRetriever()) as backends:
            return await execute(LOAN, case, "compiled", backends, plan=plan)

    record = asyncio.run(collect())
    classifier = next(n for n in record.nodes if n.node_id == "classify_segment")
    assert classifier.error is None and classifier.retries == 1
    assert "mock-small+mock-large" == classifier.usage.model
