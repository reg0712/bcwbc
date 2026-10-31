"""Parallelisation: nodes with no data path between them may run at the same time.

The executor starts each node once its inputs and order edges are done. Order edges keep a
compiled run as safe as file order around writes:
- a write waits for every node listed before it, so writes keep their order, earlier reads see
  the state before the write, and no write runs speculatively ahead of a step that may fail;
- every read or write listed after a write waits for it (read-your-writes). The compiler
  cannot tell which records a write changes, so a later read never runs before it.
Pure nodes have no outside contact, so they move freely around writes.
"""

from __future__ import annotations

from synapse.graph import build_graph, levels
from synapse.ir import Decision, Effect, Profile, Workflow


def safe_order_edges(wf: Workflow) -> list[tuple[str, str]]:
    """Order edges (before, after) for the write barriers described above, in file order."""
    edges = []
    for i, node in enumerate(wf.nodes):
        if node.effect is Effect.WRITE:
            edges.extend((prior.id, node.id) for prior in wf.nodes[:i])
            edges.extend((node.id, later.id) for later in wf.nodes[i + 1:]
                         if later.effect in (Effect.READ, Effect.WRITE))
    return list(dict.fromkeys(edges))


def parallel_levels(wf: Workflow) -> list[list[str]]:
    """Groups of nodes that may run together once data and order edges are respected."""
    graph = build_graph(wf)
    graph.add_edges_from(safe_order_edges(wf))
    return levels(graph)


class Parallelise:
    name = "parallelise"

    def plan(self, wf: Workflow, profile: Profile | None = None) -> list[Decision]:
        """An applied decision per group of nodes that may start together, and a skipped one
        per write, which keeps its place in file order."""
        decisions = [Decision(pass_name=self.name, nodes=group, status="applied",
                              reason=f"{', '.join(group)} may run together: no data path "
                                     f"or write barrier between them")
                     for group in parallel_levels(wf) if len(group) > 1]
        if not decisions:
            decisions.append(Decision(pass_name=self.name, nodes=[], status="skipped",
                                      reason="every step depends on the one before it"))
        decisions.extend(
            Decision(pass_name=self.name, nodes=[node.id], status="skipped",
                     reason=f"{node.id} not moved: a write waits for every step listed "
                            f"before it, and later reads and writes wait for it")
            for node in wf.nodes if node.effect is Effect.WRITE)
        return decisions

    def apply(self, wf: Workflow, decisions: list[Decision]) -> Workflow:
        """The workflow unchanged: parallelism comes from the plan's order edges."""
        return wf.model_copy(deep=True)
