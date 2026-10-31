"""Dependency graph (M1): the loan workflow's dependencies, cycle and unknown-input errors,
write ordering, levels and the critical path."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytest

from synapse import ROOT
from synapse.cli import main
from synapse.graph import (CycleError, GraphError, UnknownInputError, build_graph,
                           critical_path, dependencies, levels, topological_order, write_chain)
from synapse.ir import Node, Workflow
from synapse.parser import load_workflow

LOAN = ROOT / "workflows" / "loan_prescreen.yaml"
LOAN_NODES = ["parse_app", "kyc_check", "bureau_check", "fetch_statement", "statement_analysis",
              "policy_lookup_1", "policy_lookup_2", "offers_lookup", "classify_segment",
              "risk_summary", "notify_customer"]

# config/mock.yaml latencies: tool calls by API, LLM calls by model tier, retrieval as 0.
LOAN_LATENCY_MS = {
    "parse_app": 1500, "kyc_check": 800, "bureau_check": 1200, "fetch_statement": 1500,
    "statement_analysis": 1500, "offers_lookup": 600, "classify_segment": 1500,
    "risk_summary": 1500, "notify_customer": 200,
}


def toy(*nodes: dict[str, Any]) -> Workflow:
    """A workflow with input x whose nodes are pure transforms unless a dict says otherwise."""
    return Workflow(name="toy", inputs=["x"],
                    nodes=[Node(**{"kind": "transform", "effect": "pure", **raw}) for raw in nodes])


@pytest.fixture(scope="module")
def loan() -> nx.DiGraph:
    return build_graph(load_workflow(LOAN))


def test_loan_workflow_parses() -> None:
    wf = load_workflow(LOAN)
    assert wf.name == "loan_prescreen"
    assert wf.inputs == ["application_pdf", "customer_id"]
    assert [node.id for node in wf.nodes] == LOAN_NODES
    assert next(n for n in wf.nodes if n.id == "risk_summary").tests == [{"schema": "risk.json"}]


def test_risk_summary_depends_on_checks_policies_and_segment(loan: nx.DiGraph) -> None:
    assert dependencies(loan, "risk_summary") == [
        "kyc_check", "bureau_check", "statement_analysis",
        "policy_lookup_1", "policy_lookup_2", "classify_segment"]


def test_dotted_inputs_depend_on_the_producing_node(loan: nx.DiGraph) -> None:
    assert dependencies(loan, "kyc_check") == ["parse_app"]
    assert loan.edges["parse_app", "kyc_check"]["fields"] == ["applicant.pan", "applicant.name"]
    assert dependencies(loan, "notify_customer") == ["risk_summary"]


def test_workflow_inputs_add_no_dependencies(loan: nx.DiGraph) -> None:
    for node_id in ("parse_app", "fetch_statement", "policy_lookup_1", "offers_lookup"):
        assert dependencies(loan, node_id) == []


def test_levels_group_nodes_that_could_run_together(loan: nx.DiGraph) -> None:
    assert levels(loan) == [
        ["parse_app", "fetch_statement", "policy_lookup_1", "policy_lookup_2", "offers_lookup"],
        ["kyc_check", "bureau_check", "statement_analysis", "classify_segment"],
        ["risk_summary"],
        ["notify_customer"],
    ]


def test_topological_order_keeps_file_order(loan: nx.DiGraph) -> None:
    assert topological_order(loan) == LOAN_NODES


def test_cycle_raises() -> None:
    wf = toy({"id": "a", "inputs": ["b_out"], "outputs": ["a_out"]},
             {"id": "b", "inputs": ["a_out"], "outputs": ["b_out"]})
    with pytest.raises(CycleError, match="cycle"):
        build_graph(wf)


def test_reading_its_own_output_is_a_cycle() -> None:
    with pytest.raises(CycleError):
        build_graph(toy({"id": "a", "inputs": ["a_out"], "outputs": ["a_out"]}))


def test_unknown_input_raises() -> None:
    wf = toy({"id": "a", "inputs": ["ghost.field"], "outputs": ["a_out"]})
    with pytest.raises(UnknownInputError, match="a: input 'ghost.field'"):
        build_graph(wf)


@pytest.mark.parametrize(("nodes", "message"), [
    pytest.param([{"id": "a", "outputs": ["p"]}, {"id": "a", "outputs": ["q"]}],
                 "duplicate node id", id="duplicate-id"),
    pytest.param([{"id": "a", "outputs": ["p"]}, {"id": "b", "outputs": ["p"]}],
                 "produced by both a and b", id="duplicate-output"),
    pytest.param([{"id": "a", "outputs": ["x"]}], "name of a workflow input", id="shadows-input"),
    pytest.param([{"id": "b", "inputs": ["a_out"], "outputs": ["b_out"]},
                  {"id": "a", "outputs": ["a_out"]}],
                 "b is listed before a", id="consumer-before-producer"),
])
def test_malformed_graphs_raise(nodes: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(GraphError, match=message):
        build_graph(toy(*nodes))


def test_consecutive_writes_get_order_edges() -> None:
    g = build_graph(toy({"id": "w1", "inputs": ["x"], "outputs": ["o1"], "effect": "write"},
                        {"id": "p", "inputs": ["x"], "outputs": ["o2"]},
                        {"id": "w2", "inputs": ["x"], "outputs": ["o3"], "effect": "write"},
                        {"id": "w3", "inputs": ["x"], "outputs": ["o4"], "effect": "write"}))
    order_edges = sorted((src, dst) for src, dst, kinds in g.edges(data="kinds") if "order" in kinds)
    assert order_edges == [("w1", "w2"), ("w2", "w3")]
    assert write_chain(g) == ["w1", "w2", "w3"]
    assert levels(g) == [["w1", "p"], ["w2"], ["w3"]]


def test_untagged_node_is_ordered_as_a_write() -> None:
    wf = Workflow(name="toy", inputs=["x"], nodes=[
        Node(id="sms", kind="tool", tool="sms", inputs=["x"], outputs=["a"], effect="write"),
        Node(id="untagged", kind="tool", tool="crm", inputs=["x"], outputs=["b"]),
    ])
    g = build_graph(wf)
    assert "order" in g.edges["sms", "untagged"]["kinds"]


def test_write_reading_a_later_write_raises() -> None:
    wf = toy({"id": "w1", "inputs": ["o2"], "outputs": ["o1"], "effect": "write"},
             {"id": "w2", "inputs": ["x"], "outputs": ["o2"], "effect": "write"})
    with pytest.raises(CycleError):
        build_graph(wf)


def test_critical_path_from_given_latencies() -> None:
    g = build_graph(toy({"id": "a", "inputs": ["x"], "outputs": ["a_out"]},
                        {"id": "fast", "inputs": ["a_out"], "outputs": ["f"]},
                        {"id": "slow", "inputs": ["a_out"], "outputs": ["s"]},
                        {"id": "join", "inputs": ["f", "s"], "outputs": ["j"]}))
    assert critical_path(g, {"a": 100, "fast": 50, "slow": 400, "join": 10}) == (
        ["a", "slow", "join"], 510)
    assert critical_path(g, {"a": 100, "fast": 400, "slow": 400, "join": 10}) == (
        ["a", "fast", "join"], 510)


def test_critical_path_counts_missing_latencies_as_zero() -> None:
    g = build_graph(toy({"id": "a", "inputs": ["x"], "outputs": ["a_out"]}))
    assert critical_path(g, {}) == (["a"], 0.0)


def test_critical_path_of_loan_workflow(loan: nx.DiGraph) -> None:
    path, total = critical_path(loan, LOAN_LATENCY_MS)
    assert path == ["fetch_statement", "statement_analysis", "risk_summary", "notify_customer"]
    assert total == 4700


def test_compile_show_graph_prints_levels_and_write_order(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compile", str(LOAN), "--show-graph"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("loan_prescreen: 11 source nodes -> 9 compiled nodes")
    level_of = {line.split()[1]: line.split()[0]
                for line in out.splitlines() if line[:1].isdigit()}
    assert level_of["parse_app"] == "1"
    assert level_of["risk_summary"] == "3"
    assert level_of["notify_customer"] == "4"
    assert out.rstrip().endswith("write order: notify_customer")


def test_compile_reports_bad_workflows_without_a_traceback(
        tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("workflow: bad\ninput: [x]\nnodes:\n"
                    "  - {id: a, kind: transform, effect: pure, inputs: [ghost], outputs: [y]}\n")
    assert main(["compile", str(path)]) == 1
    assert "input 'ghost' is not a workflow input" in capsys.readouterr().err
