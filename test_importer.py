"""M9: import a real LangGraph graph and run its sample cases in SYNAPSE."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from synapse import ROOT
from synapse.backends import mock_backends
from synapse.backends.retrieval import KeywordRetriever
from synapse.compiler import compile_workflow
from synapse.executor import execute, final_outputs
from synapse.importers.langgraph_importer import import_langgraph
from synapse.validator import validate


def test_hotpot_import_and_parallel_output_preservation() -> None:
    imported = import_langgraph(str(ROOT / "workflows" / "research_agent_langgraph.py"))
    assert len(imported.samples) == 20
    assert len(imported.workflow.nodes) == 6
    assert imported.workflow.outputs == ["answer"]
    plan = compile_workflow(imported.workflow, passes=("parallelise",))
    plan.order_edges.extend(imported.control_edges)

    async def collect():
        async with mock_backends(0, retriever=KeywordRetriever(),
                                 functions=imported.functions) as backends:
            baseline = [await execute(imported.workflow, case, "baseline", backends)
                        for case in imported.samples]
            compiled = [await execute(imported.workflow, case, "compiled", backends, plan=plan)
                        for case in imported.samples]
        return baseline, compiled

    golden, candidate = asyncio.run(collect())
    checks = validate(golden, candidate, 0)
    assert checks and all(check.passed for check in checks)
    assert all(final_outputs(a)["answer"] == final_outputs(b)["answer"]
               for a, b in zip(golden, candidate))
    nodes = {n.node_id: n for n in candidate[0].nodes}
    assert nodes["retrieve_one"].start_s < nodes["retrieve_two"].end_s
    assert nodes["retrieve_two"].start_s < nodes["retrieve_one"].end_s


def test_explicit_model_metadata_does_not_invoke_functions_during_import(monkeypatch) -> None:
    from workflows import research_llm_langgraph as module

    def forbidden(_state):
        raise AssertionError("import performed work")

    monkeypatch.setattr(module, "NODES", {name: forbidden for name in module.NODES})
    monkeypatch.setattr("synapse.importers.langgraph_importer._load_module", lambda _: module)
    imported = import_langgraph("declared")
    assert len(imported.samples) == 20
    assert [n.kind.value for n in imported.workflow.nodes] == ["transform", "transform", "llm"]
    plan = compile_workflow(imported.workflow)
    assert any(d.pass_name == "parallelise" and d.status == "applied" for d in plan.decisions)


def test_native_and_imported_model_research_agree_without_network() -> None:
    from workflows import research_llm_langgraph as module

    imported = import_langgraph(str(ROOT / "workflows" / "research_llm_langgraph.py"))

    async def compare():
        async with mock_backends(.001, retriever=KeywordRetriever(),
                                 functions=imported.functions) as backends:
            graph = module.build_graph(backends.llm).compile()
            for case in imported.samples:
                native = await graph.ainvoke(case)
                compiled = await execute(imported.workflow, case, "compiled", backends)
                assert not any(n.error for n in compiled.nodes)
                assert native["answer"] == final_outputs(compiled)["answer"]
    asyncio.run(compare())


def test_importer_discovers_data_parallelism_in_serial_langgraph(monkeypatch) -> None:
    class State(TypedDict, total=False):
        seed: int
        a: int
        b: int
        c: int
        d: int

    def a(state):
        return {"a": state["seed"] + 1}

    def b(state):
        time.sleep(0.02)
        return {"b": state["a"] * 2}

    def c(state):
        time.sleep(0.02)
        return {"c": state["a"] * 3}

    def d(state):
        return {"d": state["b"] + state["c"]}

    functions = {"A": a, "B": b, "C": c, "D": d}

    def graph():
        g = StateGraph(State)
        for name, function in functions.items():
            g.add_node(name, function)
        g.add_edge(START, "A")
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("D", END)
        return g

    module = SimpleNamespace(build_graph=graph, NODES=functions,
                             EFFECTS={name: "pure" for name in functions},
                             SAMPLE_INPUTS=[{"case_id": "S01", "seed": 2}], OUTPUTS=["d"])
    monkeypatch.setattr("synapse.importers.langgraph_importer._load_module", lambda _: module)
    imported = import_langgraph("toy")
    assert {node.id: node.inputs for node in imported.workflow.nodes} == {
        "A": ["seed"], "B": ["a"], "C": ["a"], "D": ["b", "c"]}
    assert imported.control_edges == []
    plan = compile_workflow(imported.workflow, passes=("parallelise",))

    async def collect():
        async with mock_backends(0, retriever=KeywordRetriever(),
                                 functions=imported.functions) as backends:
            baseline = await execute(imported.workflow, imported.samples[0], "baseline", backends)
            compiled = await execute(imported.workflow, imported.samples[0], "compiled",
                                     backends, plan=plan)
        return baseline, compiled

    baseline, compiled = asyncio.run(collect())
    assert final_outputs(baseline)["d"] == final_outputs(compiled)["d"] == 15
    nodes = {node.node_id: node for node in compiled.nodes}
    assert nodes["B"].start_s < nodes["C"].end_s
    assert nodes["C"].start_s < nodes["B"].end_s

    module.EFFECTS.pop("B")
    with pytest.raises(ValueError, match="write nodes need explicit I/O metadata"):
        import_langgraph("toy")


def test_conditional_edges_become_required_gates(monkeypatch) -> None:
    class State(TypedDict, total=False):
        choice: bool
        a: bool
        b: str
        c: str

    functions = {
        "A": lambda state: {"a": state["choice"]},
        "B": lambda state: {"b": "left" if state["a"] else "right"},
        "C": lambda state: {"c": "right" if state["a"] else "left"},
    }

    def graph():
        g = StateGraph(State)
        for name, function in functions.items():
            g.add_node(name, function)
        g.add_edge(START, "A")
        g.add_conditional_edges("A", lambda state: "B" if state["a"] else "C",
                                {"B": "B", "C": "C"})
        g.add_edge("B", END)
        g.add_edge("C", END)
        return g

    module = SimpleNamespace(build_graph=graph, NODES=functions,
                             EFFECTS={name: "pure" for name in functions},
                             SAMPLE_INPUTS=[{"case_id": "S01", "choice": True}], OUTPUTS=["a"])
    monkeypatch.setattr("synapse.importers.langgraph_importer._load_module", lambda _: module)
    imported = import_langgraph("conditional")
    assert imported.conditional_gates == ["gate_A"]
    assert imported.control_edges == [("A", "gate_A"), ("gate_A", "B"), ("gate_A", "C")]
    assert next(node for node in imported.workflow.nodes if node.id == "gate_A").required
