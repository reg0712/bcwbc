"""Dead-node elimination: remove pure and read nodes whose results nothing uses.

A node stays when it is a write (a node without an effect tag counts as one), is marked
required, produces a declared workflow output, or feeds a node that stays. Everything else is
removed, which reaches the fixed point in one pass: a producer whose only readers are removed
is removed too. Without declared outputs the final results are unknown, so nothing is removed.
"""

from __future__ import annotations

import networkx as nx

from synapse.graph import build_graph, consumers, input_root
from synapse.ir import Decision, Effect, Node, Profile, Workflow


def _keep_reason(node: Node, published: set[str]) -> str | None:
    """Why a node must stay even if no other node reads its result."""
    if node.effect is Effect.WRITE:
        return "write effect"
    if node.required:
        return "required check"
    fields = [field for field in node.outputs if field in published]
    if fields:
        return f"produces workflow output {', '.join(fields)}"
    return None


def _live(wf: Workflow, g: nx.DiGraph) -> set[str]:
    """Nodes that must stay, plus every node whose result they read, directly or not."""
    published = {input_root(field) for field in wf.outputs}
    roots = {node.id for node in wf.nodes if _keep_reason(node, published)}
    live = set(roots)
    for node_id in roots:
        live |= nx.ancestors(g, node_id)
    return live


class DeadNodes:
    name = "dead_nodes"

    def plan(self, wf: Workflow, profile: Profile | None = None) -> list[Decision]:
        """An applied decision per removable node, and a skipped one per node that nothing
        reads but that must stay (a write or a required check)."""
        if not wf.outputs:
            return [Decision(pass_name=self.name, nodes=[], status="skipped",
                             reason="workflow declares no outputs, so its final results are "
                                    "unknown and no node is provably dead")]
        g = build_graph(wf)
        live = _live(wf, g)
        published = {input_root(field) for field in wf.outputs}
        decisions = []
        for node in wf.nodes:
            if node.id not in live:
                unused = ", ".join(node.outputs) or "its result"
                decisions.append(Decision(
                    pass_name=self.name, nodes=[node.id], status="applied",
                    reason=f"{node.id} removed: nothing that stays reads {unused}, and it is "
                           f"a {node.effect.value} node, not a workflow output"))
                continue
            read = any(reader in live for reader in consumers(g, node.id))
            if not read and not any(field in published for field in node.outputs):
                decisions.append(Decision(
                    pass_name=self.name, nodes=[node.id], status="skipped",
                    reason=f"{node.id} kept although nothing reads its result: "
                           f"{_keep_reason(node, published)}"))
        return decisions

    def apply(self, wf: Workflow, decisions: list[Decision]) -> Workflow:
        """Remove the nodes of applied decisions; refuses writes, required checks, producers
        of workflow outputs and producers that a remaining node still reads."""
        removed = {node_id for decision in decisions if decision.status == "applied"
                   for node_id in decision.nodes}
        published = {input_root(field) for field in wf.outputs}
        by_id = {node.id: node for node in wf.nodes}
        for node_id in sorted(removed):
            if node_id not in by_id:
                raise ValueError(f"cannot remove {node_id}: no such node")
            reason = _keep_reason(by_id[node_id], published)
            if reason:
                raise ValueError(f"cannot remove {node_id}: {reason}")
        result = wf.model_copy(update={"nodes": [n for n in wf.nodes if n.id not in removed]},
                               deep=True)
        gone = {field for node_id in removed for field in by_id[node_id].outputs}
        for node in result.nodes:
            still_read = [field for field in node.inputs if input_root(field) in gone]
            if still_read:
                raise ValueError(f"cannot remove the producer of {still_read[0]}: "
                                 f"{node.id} still reads it")
        return result
