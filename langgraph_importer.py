"""Import safe LangGraph DAGs; trace state keys without LangGraph scheduling runs."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from synapse.graph import build_graph
from synapse.ir import Effect, Node, Workflow
from synapse.parser import parse_workflow


class TracingState(dict):
    def __init__(self, values: dict[str, Any]):
        super().__init__(values)
        self.reads: set[str] = set()

    def __getitem__(self, key):
        self.reads.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.reads.add(key)
        return super().get(key, default)


@dataclass
class ImportedWorkflow:
    workflow: Workflow
    functions: dict[str, Any]
    control_edges: list[tuple[str, str]]
    samples: list[dict[str, Any]]
    conditional_gates: list[str] | None = None


def _load_module(name_or_path: str):
    path = Path(name_or_path)
    if path.suffix == ".py" and path.exists():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(name_or_path)


def import_langgraph(name_or_path: str) -> ImportedWorkflow:
    module = _load_module(name_or_path)
    for required in ("build_graph", "NODES", "EFFECTS", "SAMPLE_INPUTS"):
        if not hasattr(module, required):
            raise ValueError(f"{name_or_path} must export {required}")
    drawn = module.build_graph().compile().get_graph()
    functions = dict(module.NODES)
    names = [id for id in drawn.nodes if id not in ("__start__", "__end__")]
    if set(names) != set(functions):
        raise ValueError("NODES must match the compiled LangGraph nodes")
    source_edges = [(edge.source, edge.target) for edge in drawn.edges
                    if edge.source in functions and edge.target in functions]
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    graph.add_edges_from(source_edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("LangGraph control graph is cyclic")
    order = list(nx.topological_sort(graph))
    samples = list(module.SAMPLE_INPUTS)
    if not samples:
        raise ValueError("SAMPLE_INPUTS needs at least one sample")
    sample = samples[0]
    specifications = getattr(module, "NODE_SPECS", None)
    if specifications is not None:
        if set(specifications) != set(names):
            raise ValueError("NODE_SPECS must describe every graph node")
        declared = parse_workflow({"workflow": getattr(module, "WORKFLOW_NAME", Path(name_or_path).stem),
                                   "input": [key for key in sample if key != "case_id"],
                                   "output": list(getattr(module, "OUTPUTS", [])),
                                   "nodes": [{**specifications[id], "id": id} for id in order]})
        for node in declared.nodes:
            if node.effect != Effect(module.EFFECTS.get(node.id, "write")):
                raise ValueError(f"{node.id}: NODE_SPECS effect differs from EFFECTS")
    state = {key: value for key, value in sample.items() if key != "case_id"}
    produced: set[str] = set()
    nodes: dict[str, Node] = {}
    for id in order:
        if specifications is not None:
            nodes[id] = next(node for node in declared.nodes if node.id == id)
            continue
        effect = Effect(module.EFFECTS.get(id, "write"))
        if effect is Effect.WRITE:
            raise ValueError(f"{id}: write nodes need explicit I/O metadata; tracing would perform a write")
        if effect is Effect.READ:
            raise ValueError(f"{id}: read nodes need explicit I/O metadata; importing must not call services")
        tracing = TracingState(state)
        value = functions[id](tracing)
        if inspect.isawaitable(value):
            raise ValueError(f"{id}: async trace functions need explicit I/O metadata")
        if not isinstance(value, dict):
            raise ValueError(f"{id}: function must return a state update dict")
        if any(key in produced or key in sample for key in value):
            raise ValueError(f"{id}: output overwrites an existing state key")
        nodes[id] = Node(id=id, kind="transform", tool=id, effect=effect,
                         deterministic=True, inputs=sorted(tracing.reads),
                         outputs=list(value))
        state.update(value)
        produced.update(value)
    # Ordinary LangGraph edges describe the original schedule. Once the state reads have
    # been traced, preserving those edges would prevent safe parallelism (A -> B -> C,
    # where B and C both read only A). Conditional edges are different: retain them as
    # explicit gates so callers can inspect the branch, and refuse execution until the
    # executor learns to select exactly one outgoing edge.
    conditional: dict[str, list[str]] = {}
    for edge in drawn.edges:
        if edge.conditional and edge.source in functions and edge.target in functions:
            conditional.setdefault(edge.source, []).append(edge.target)
    control: list[tuple[str, str]] = []
    for source, targets in conditional.items():
        gate = f"gate_{source}"
        if gate in nodes:
            raise ValueError(f"{source}: generated gate name {gate!r} collides with a node")
        nodes[gate] = Node(id=gate, kind="gate", effect="pure", required=True,
                           inputs=[], outputs=[])
        control.append((source, gate))
        control.extend((gate, target) for target in dict.fromkeys(targets))
    for source, target in source_edges:
        if source in conditional or target in conditional:
            continue
        if nodes[source].effect is Effect.WRITE or nodes[target].effect is Effect.WRITE:
            control.append((source, target))
    scheduled = nx.DiGraph()
    scheduled.add_nodes_from(nodes)
    scheduled.add_edges_from(control)
    if not nx.is_directed_acyclic_graph(scheduled):
        raise ValueError("imported control graph is cyclic")
    sorted_nodes = [nodes[id] for id in nx.topological_sort(scheduled)]
    inputs = [key for key in sample if key != "case_id"]
    outputs = list(getattr(module, "OUTPUTS", []))
    if not outputs:
        terminal = [id for id in order if graph.out_degree(id) == 0]
        outputs = [field for node in sorted_nodes if node.id in terminal for field in node.outputs]
    wf = Workflow(name=getattr(module, "WORKFLOW_NAME", Path(name_or_path).stem),
                  inputs=inputs, nodes=sorted_nodes, outputs=outputs)
    build_graph(wf)
    return ImportedWorkflow(wf, functions, control, samples,
                            [f"gate_{source}" for source in conditional])
