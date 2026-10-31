"""Dependency graph (NetworkX) for a workflow.

Data edges run from the one node that produces an input to each node that reads it; order
edges keep write nodes in file order. Also here: topological order, parallel levels and the
latency-weighted critical path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import networkx as nx

from synapse.ir import Effect, Workflow


class GraphError(ValueError):
    """Dependencies that cannot form a valid execution graph."""


class UnknownInputError(GraphError):
    """An input that is neither a workflow input nor any node's output."""


class CycleError(GraphError):
    """Dependencies that loop back on themselves."""


def input_root(field: str) -> str:
    """The first segment of an input reference: 'applicant' for 'applicant.pan'."""
    return field.split(".", 1)[0]


def build_graph(wf: Workflow) -> nx.DiGraph:
    """Build the dependency graph of a workflow.

    Each graph node carries its Node model ("node") and file position ("index"); each edge
    carries its kinds ("data", "order") and the input fields it feeds. Raises GraphError on
    duplicate ids or outputs and on a workflow output no node produces, UnknownInputError on
    an input nobody produces, and CycleError on a loop.
    """
    g = nx.DiGraph(workflow=wf.name)
    producers = _add_nodes(g, wf)
    for field in wf.outputs:
        if input_root(field) not in producers:
            raise GraphError(f"workflow output {field!r} is not produced by any node")
    for node in wf.nodes:
        for field in node.inputs:
            root = input_root(field)
            if root in wf.inputs:
                continue
            if root not in producers:
                raise UnknownInputError(f"{node.id}: input {field!r} is not a workflow input "
                                        f"or any node's output")
            _link(g, producers[root], node.id, "data", field)
    writes = [node.id for node in wf.nodes if node.effect is Effect.WRITE]
    for earlier, later in zip(writes, writes[1:]):
        _link(g, earlier, later, "order")
    _check_acyclic(g)
    _check_file_order(g)
    return g


def _add_nodes(g: nx.DiGraph, wf: Workflow) -> dict[str, str]:
    """Add every node to g and map each output field to the node that produces it."""
    producers: dict[str, str] = {}
    for index, node in enumerate(wf.nodes):
        if node.id in g:
            raise GraphError(f"duplicate node id {node.id!r}")
        g.add_node(node.id, node=node, index=index)
        for field in node.outputs:
            if field in wf.inputs:
                raise GraphError(f"{node.id}: output {field!r} has the name of a workflow input")
            if field in producers:
                raise GraphError(f"output {field!r} is produced by both {producers[field]} "
                                 f"and {node.id}")
            producers[field] = node.id
    return producers


def _link(g: nx.DiGraph, src: str, dst: str, kind: str, field: str | None = None) -> None:
    """Add or extend the edge src -> dst with a kind and, for data edges, the field it feeds."""
    if not g.has_edge(src, dst):
        g.add_edge(src, dst, kinds=set(), fields=[])
    edge = g.edges[src, dst]
    edge["kinds"].add(kind)
    if field is not None:
        edge["fields"].append(field)


def _check_acyclic(g: nx.DiGraph) -> None:
    """Raise CycleError naming one loop if the graph has any."""
    try:
        cycle = nx.find_cycle(g)
    except nx.NetworkXNoCycle:
        return
    path = " -> ".join([src for src, _ in cycle] + [cycle[0][0]])
    raise CycleError(f"dependency cycle: {path}")


def _check_file_order(g: nx.DiGraph) -> None:
    """Raise GraphError if a node is listed before one it depends on: baseline mode runs
    nodes in file order, so every producer must come first."""
    for src, dst in g.edges:
        if g.nodes[src]["index"] > g.nodes[dst]["index"]:
            raise GraphError(f"{dst} is listed before {src}, which it depends on; "
                             f"baseline mode runs nodes in file order")


def _position(g: nx.DiGraph) -> Callable[[str], int]:
    """Sort key: a node's position in the workflow file."""
    return lambda node_id: g.nodes[node_id]["index"]


def topological_order(g: nx.DiGraph) -> list[str]:
    """Node ids in dependency order, ties broken by file position."""
    return list(nx.lexicographical_topological_sort(g, key=_position(g)))


def levels(g: nx.DiGraph) -> list[list[str]]:
    """Groups of nodes that could run at the same time: each depends only on earlier groups."""
    return [sorted(generation, key=_position(g)) for generation in nx.topological_generations(g)]


def dependencies(g: nx.DiGraph, node_id: str) -> list[str]:
    """Nodes whose outputs node_id reads directly, in file order; order-only edges excluded."""
    producers = [src for src in g.predecessors(node_id) if "data" in g.edges[src, node_id]["kinds"]]
    return sorted(producers, key=_position(g))


def consumers(g: nx.DiGraph, node_id: str) -> list[str]:
    """Nodes that read node_id's outputs directly, in file order; order-only edges excluded."""
    readers = [dst for dst in g.successors(node_id) if "data" in g.edges[node_id, dst]["kinds"]]
    return sorted(readers, key=_position(g))


def write_chain(g: nx.DiGraph) -> list[str]:
    """Write nodes in the order they must keep."""
    return [node_id for node_id in topological_order(g)
            if g.nodes[node_id]["node"].effect is Effect.WRITE]


def critical_path(g: nx.DiGraph, latency_ms: Mapping[str, float]) -> tuple[list[str], float]:
    """The longest latency-weighted chain of dependent nodes and its length in milliseconds.

    No amount of parallelism finishes a run faster than this chain. Nodes missing from
    latency_ms count as 0 ms; ties go to the node listed first in the file.
    """
    position = _position(g)
    finish: dict[str, float] = {}
    via: dict[str, str | None] = {}
    for node_id in topological_order(g):
        before = max(g.predecessors(node_id), default=None,
                     key=lambda src: (finish[src], -position(src)))
        via[node_id] = before
        start = finish[before] if before is not None else 0.0
        finish[node_id] = start + float(latency_ms.get(node_id, 0.0))
    if not finish:
        return [], 0.0
    end: str | None = max(finish, key=lambda node_id: (finish[node_id], -position(node_id)))
    total = finish[end]
    path = []
    while end is not None:
        path.append(end)
        end = via[end]
    return path[::-1], total
