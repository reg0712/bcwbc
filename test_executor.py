"""Executor (M2): baseline mode runs every node once, in file order, and reaches the test
rule's decision on all 20 cases, sending exactly one SMS per run."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

import pytest

from mock_services import MOCK_TASKS, suite_for
from mock_services.bank_api import create_app
from synapse import ROOT
from synapse.backends import mock_backends
from synapse.backends.llm import MockLLM
from synapse.backends.retrieval import KeywordRetriever
from synapse.cli import main
from synapse.config import load_config
from synapse.executor import execute, final_outputs, resolve
from synapse.ir import Node, RunRecord, Workflow
from synapse.parser import load_workflow
from synapse.compiler import compile_workflow
from synapse.passes.memoise import ExactCache

SCALE = 0.001
LOAN_PATH = ROOT / "workflows" / "loan_prescreen.yaml"
LOAN = load_workflow(LOAN_PATH)
SUITE = suite_for("loan_prescreen")


def run_cases(cases: list[dict[str, Any]], wf: Workflow = LOAN,
              mode: str = "baseline") -> tuple[list[RunRecord], list[dict[str, Any]]]:
    """Run cases one after another on fresh mock backends; returns the runs and sent SMS."""
    async def run_all() -> list[RunRecord]:
        async with mock_backends(SCALE, bank_app=bank, retriever=KeywordRetriever()) as backends:
            return [await execute(wf, case, mode, backends) for case in cases]  # type: ignore[arg-type]
    bank = create_app(scale=SCALE)
    return asyncio.run(run_all()), bank.state.sms_log


def test_baseline_runs_every_node_once_in_file_order() -> None:
    (record,), _ = run_cases([SUITE.case("A01")])
    assert [node.node_id for node in record.nodes] == [node.id for node in LOAN.nodes]
    assert all(node.error is None for node in record.nodes)
    for earlier, later in zip(record.nodes, record.nodes[1:]):
        assert later.start_s >= earlier.end_s
    assert (record.workflow, record.mode, record.case_id) == ("loan_prescreen", "baseline", "A01")
    assert record.nodes[-1].end_s * 1000 <= record.wall_ms


def test_baseline_reaches_the_rule_decision_on_all_20_cases() -> None:
    records, _ = run_cases(SUITE.cases())
    decisions = {r.case_id: resolve(SUITE.key_field, final_outputs(r)) for r in records}
    assert decisions == {case_id: SUITE.expected(case_id) for case_id in decisions}
    assert Counter(decisions.values()) == {"approve": 8, "refer": 7, "decline": 5}


def test_each_run_sends_exactly_one_sms_with_its_decision() -> None:
    _, sms_log = run_cases([SUITE.case("A03"), SUITE.case("A04")])
    assert [sent["customer_id"] for sent in sms_log] == ["CUST-A03", "CUST-A04"]
    assert "refer" in sms_log[0]["message"]
    assert "decline" in sms_log[1]["message"]


def test_llm_nodes_report_tokens_and_tools_report_none() -> None:
    (record,), _ = run_cases([SUITE.case("A01")])
    usage = {node.node_id: node.usage for node in record.nodes}
    assert usage["risk_summary"].tokens_in > usage["classify_segment"].tokens_in > 0
    assert usage["risk_summary"].model == "mock-large"
    assert usage["kyc_check"].tokens_in == usage["kyc_check"].tokens_out == 0


def test_missing_workflow_input_raises() -> None:
    with pytest.raises(ValueError, match="customer_id"):
        run_cases([{"case_id": "X", "application_pdf": "..."}])


def test_failing_node_stops_the_run_and_is_recorded() -> None:
    wf = Workflow(name="toy", inputs=["customer_id"], nodes=[
        Node(id="bad", kind="tool", tool="teleport_api", inputs=["customer_id"],
             outputs=["x"], effect="read"),
        Node(id="after", kind="tool", tool="offers_api", inputs=["customer_id"],
             outputs=["y"], effect="read"),
    ])
    (record,), _ = run_cases([{"case_id": "T1", "customer_id": "CUST-A01"}], wf=wf)
    assert [node.node_id for node in record.nodes] == ["bad"]
    assert "unknown tool 'teleport_api'" in str(record.nodes[0].error)


def test_compiled_mode_matches_all_20_cases_and_preserves_one_sms_per_case() -> None:
    records, sms_log = run_cases(SUITE.cases(), mode="compiled")
    assert len(sms_log) == 20
    assert all(r.mode == "compiled" and all(n.error is None for n in r.nodes) for r in records)
    assert {r.case_id: resolve(SUITE.key_field, final_outputs(r)) for r in records} == {
        case["case_id"]: SUITE.expected(case["case_id"]) for case in SUITE.cases()}
    assert all("offers_lookup" not in [n.node_id for n in r.nodes] for r in records)
    assert all("policy_lookup_2" not in [n.node_id for n in r.nodes] for r in records)


def test_resolve_reads_nested_fields_and_names_what_is_missing() -> None:
    assert resolve("applicant.pan", {"applicant": {"pan": "P"}}) == "P"
    with pytest.raises(KeyError, match="age"):
        resolve("applicant.age", {"applicant": {"pan": "P"}})


def test_cli_run_prints_the_decision_and_wall_time(
        embedding_model: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(LOAN_PATH), "--case", "A04", "--latency-scale", str(SCALE)]) == 0
    out = capsys.readouterr().out
    assert "notify_customer" in out
    assert out.rstrip().splitlines()[-1].split()[:2] == ["A04", "decline"]


def test_compiled_run_overlaps_independent_nodes_and_is_faster() -> None:
    (baseline,), _ = run_cases([SUITE.case("A01")])
    (compiled,), _ = run_cases([SUITE.case("A01")], mode="compiled")
    by_id = {node.node_id: node for node in compiled.nodes}
    assert by_id["fetch_statement"].start_s < by_id["parse_app"].end_s
    assert compiled.wall_ms < baseline.wall_ms
    assert sum(n.usage.cost for n in compiled.nodes) < sum(n.usage.cost for n in baseline.nodes)


def test_cli_run_rejects_an_unknown_case(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(LOAN_PATH), "--case", "Z99"]) == 1
    assert "cases are A01" in capsys.readouterr().err


def test_shared_cache_never_crosses_provider_or_bank_context() -> None:
    case = SUITE.case("A01")
    plan = compile_workflow(LOAN)
    cache = ExactCache()

    async def collect():
        async with mock_backends(SCALE, retriever=KeywordRetriever()) as first:
            original = await execute(LOAN, case, "compiled", first, plan=plan, cache=cache)
            repeat = await execute(LOAN, case, "compiled", first, plan=plan, cache=cache)
        async with mock_backends(SCALE, retriever=KeywordRetriever()) as second:
            second.provider = "gemini"  # mock adapter, different provider namespace
            names = load_config("models.yaml")["providers"]["gemini"]
            second.llm = MockLLM(MOCK_TASKS, load_config("mock.yaml")["llm_latency_ms"],
                                 names, SCALE)
            second.model_names = {"small": names["small"], "large": names["large"]}
            other = await execute(LOAN, case, "compiled", second, plan=plan, cache=cache)
        return original, repeat, other

    original, repeat, other = asyncio.run(collect())
    def node(record: RunRecord, name: str):
        return next(item for item in record.nodes if item.node_id == name)

    assert not node(original, "parse_app").cache_hit
    assert node(repeat, "parse_app").cache_hit
    assert not node(other, "parse_app").cache_hit
    assert not node(other, "bureau_check").cache_hit
    assert not other.cold_cache
