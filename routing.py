"""Model routing: move checked LLM steps that are not the final decision to the small model.

A routed node runs on the small model and escalates to the large one when its schema test
fails (the cascade in cascade.py). Routing is legal only for pure LLM nodes on the large model
that have a checkable test and are not a final decision step: an LLM step whose result reaches
a workflow output or a write without passing through another LLM step. When a workflow
declares no outputs, results that nothing reads count as outputs.

The node's tests must also pass on the small model across the test set, which only a run can
show. Without evidence routing stays off. A validator can explicitly request a probe plan,
which it must test before accepting; with evidence only nodes that passed move.
"""

from __future__ import annotations

from collections.abc import Mapping

from synapse.graph import build_graph, consumers, input_root, topological_order
from synapse.ir import Decision, Effect, Node, NodeKind, Profile, Workflow
from synapse.passes.cascade import checkable


def final_steps(wf: Workflow) -> dict[str, str]:
    """The final decision steps of a workflow, each with the output or write it reaches."""
    g = build_graph(wf)
    published = {input_root(field) for field in wf.outputs}
    reaches: dict[str, str | None] = {}
    for node_id in reversed(topological_order(g)):
        node: Node = g.nodes[node_id]["node"]
        readers = consumers(g, node_id)
        reached = next((f"workflow output {field}" for field in node.outputs
                        if field in published), None)
        if reached is None and not wf.outputs and not readers:
            reached = "a final result (the workflow declares no outputs)"
        for reader_id in readers:
            if reached is not None:
                break
            reader: Node = g.nodes[reader_id]["node"]
            if reader.effect is Effect.WRITE:
                reached = f"write {reader_id}"
            elif reader.kind is not NodeKind.LLM:
                reached = reaches[reader_id]
        reaches[node_id] = reached
    return {node.id: reach for node in wf.nodes
            if node.kind is NodeKind.LLM and (reach := reaches[node.id]) is not None}


def route_blocker(node: Node, final: Mapping[str, str]) -> str | None:
    """Why a node must stay on its model, or None when routing it is legal."""
    if node.kind is not NodeKind.LLM:
        return "not an LLM step"
    if node.model != "large":
        return "already on the small model" if node.model == "small" else "no model tier"
    if node.effect is not Effect.PURE:
        return f"{node.effect.value} effect"
    if not checkable(node):
        return "no schema test to catch a wrong small-model answer"
    if node.id in final:
        return f"final decision step: its result reaches {final[node.id]}"
    return None


class Routing:
    name = "routing"

    def __init__(self, evidence: Mapping[str, str | None] | None = None, *,
                 probe: bool = False) -> None:
        """evidence maps a node id to None when its tests passed on the small model for every
        test case, or to what failed; probe allows a validation-only trial without evidence."""
        self.evidence = evidence
        self.probe = probe

    def plan(self, wf: Workflow, profile: Profile | None = None) -> list[Decision]:
        """One decision per LLM node: applied, skipped (with the blocker) or rejected (when
        the evidence shows its tests failing on the small model)."""
        final = final_steps(wf)
        decisions = []
        for node in wf.nodes:
            if node.kind is not NodeKind.LLM:
                continue
            blocker = route_blocker(node, final)
            failure = self.evidence.get(node.id) if self.evidence is not None else None
            if blocker:
                status, reason = "skipped", f"{node.id} not routed: {blocker}"
            elif self.evidence is None and not self.probe:
                status, reason = "skipped", (f"{node.id} not routed: no small-model results "
                                             f"on the test set yet")
            elif self.evidence is not None and node.id not in self.evidence:
                status, reason = "skipped", (f"{node.id} not routed: no small-model results "
                                             f"on the test set yet")
            elif failure is not None:
                status, reason = "rejected", f"{node.id} not routed: {failure}"
            else:
                proof = ("its tests passed on every test case" if self.evidence is not None
                         else "trial only: validate on the test set before use")
                status, reason = "applied", (f"{node.id} routed to the small model, escalating "
                                             f"to the large one when its schema test fails; "
                                             f"{proof}")
            decisions.append(Decision(pass_name=self.name, nodes=[node.id], status=status,
                                      reason=reason))
        return decisions

    def apply(self, wf: Workflow, decisions: list[Decision]) -> Workflow:
        """Move each applied node to the small model with the large one as its fallback,
        checking legality again; raises ValueError on an illegal route."""
        selected = {node_id for decision in decisions if decision.status == "applied"
                    for node_id in decision.nodes}
        final = final_steps(wf)
        result = wf.model_copy(deep=True)
        for node in result.nodes:
            if node.id not in selected:
                continue
            blocker = route_blocker(node, final)
            if blocker:
                raise ValueError(f"cannot route {node.id}: {blocker}")
            node.model, node.fallback_model = "small", "large"
            selected.discard(node.id)
        if selected:
            raise ValueError(f"cannot route {sorted(selected)[0]}: no such node")
        return result
