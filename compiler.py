"""Apply optimisation passes and assemble a checked, executable plan.

Passes always run in the spec's table order (dead nodes, CSE, parallelise, memoise, routing),
whatever order the caller lists them in, so each pass and the scheduler see the smallest
graph. After every pass the compiler re-checks the rewrite and stops with ValueError when a
pass broke a guarantee; an unknown pass name also fails closed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping

from synapse.graph import GraphError, build_graph
from synapse.ir import Decision, Effect, ExecPlan, Profile, Workflow
from synapse.passes.base import Pass
from synapse.passes.cse import CSE
from synapse.passes.compact_inputs import CompactInputs
from synapse.passes.dead_nodes import DeadNodes
from synapse.passes.memoise import Memoise, cache_blocker
from synapse.passes.parallelise import Parallelise, safe_order_edges
from synapse.passes.routing import Routing

PASS_ORDER = ("dead_nodes", "cse", "compact_inputs", "parallelise", "memoise", "routing")
DEFAULT_PASSES = ("dead_nodes", "cse", "compact_inputs", "parallelise", "memoise")


def compile_workflow(wf: Workflow, profile: Profile | None = None, *,
                     passes: Collection[str] = DEFAULT_PASSES,
                     routing_evidence: Mapping[str, str | None] | None = None,
                     routing_probe: bool = False) -> ExecPlan:
    """Rewrite a workflow with the selected passes and return the checked plan.

    routing_evidence maps LLM node ids to None when their tests passed on the small model
    across the test set, or to what failed. Without evidence, routing stays off unless a
    validator explicitly requests a trial with routing_probe. The same workflow, passes and
    evidence always give the same plan_id.
    """
    selected = ordered_passes(passes)
    build_graph(wf)
    registry: dict[str, Pass] = {p.name: p for p in (
        DeadNodes(), CSE(), CompactInputs(), Parallelise(), Memoise(),
        Routing(routing_evidence, probe=routing_probe))}
    rewritten = wf.model_copy(deep=True)
    decisions: list[Decision] = []
    for name in selected:
        proposed = registry[name].plan(rewritten, profile)
        candidate = registry[name].apply(rewritten, proposed)
        check_rewrite(rewritten, candidate, name)
        rewritten = candidate
        decisions.extend(proposed)
    cached = list(dict.fromkeys(node_id for decision in decisions
                                if decision.pass_name == "memoise"
                                and decision.status == "applied" for node_id in decision.nodes))
    check_cached(rewritten, cached)
    order_edges = safe_order_edges(rewritten)
    return ExecPlan(plan_id=_plan_id(rewritten, order_edges, cached, selected),
                    workflow=rewritten, order_edges=order_edges, cached_nodes=cached,
                    decisions=decisions)


def ordered_passes(passes: Collection[str]) -> tuple[str, ...]:
    """The selected pass names in table order; ValueError names an unknown one."""
    if isinstance(passes, str):
        raise ValueError("passes must be a collection of pass names, not one string")
    unknown = [name for name in passes if name not in PASS_ORDER]
    if unknown:
        raise ValueError(f"unknown pass {unknown[0]!r}; known passes are {', '.join(PASS_ORDER)}")
    return tuple(name for name in PASS_ORDER if name in passes)


def check_rewrite(before: Workflow, after: Workflow, name: str) -> None:
    """Raise ValueError unless a pass kept every guarantee: the graph is still acyclic, every
    input and workflow output is still produced before use, no node was added or changed its
    kind or effect, required checks remain, and every write remains, unchanged apart from
    rewired inputs, in its original order."""
    try:
        build_graph(after)
    except GraphError as exc:
        raise ValueError(f"{name} broke the dependency graph: {exc}") from exc
    if (after.name, after.inputs, after.outputs) != (before.name, before.inputs, before.outputs):
        raise ValueError(f"{name} changed the workflow's name, inputs or outputs")
    old = {node.id: node for node in before.nodes}
    for node in after.nodes:
        if node.id not in old:
            raise ValueError(f"{name} added node {node.id}")
        if (node.kind, node.effect) != (old[node.id].kind, old[node.id].effect):
            raise ValueError(f"{name} changed the kind or effect of {node.id}")
    kept = {node.id for node in after.nodes}
    lost = [node.id for node in before.nodes if node.required and node.id not in kept]
    if lost:
        raise ValueError(f"{name} removed required check {lost[0]}")
    writes = [node for node in before.nodes if node.effect is Effect.WRITE]
    if [node.id for node in after.nodes if node.effect is Effect.WRITE] != [w.id for w in writes]:
        raise ValueError(f"{name} removed or reordered a write")
    new = {node.id: node for node in after.nodes}
    for write in writes:
        if new[write.id].model_dump(exclude={"inputs"}) != write.model_dump(exclude={"inputs"}):
            raise ValueError(f"{name} changed write {write.id}")


def check_cached(wf: Workflow, cached: list[str]) -> None:
    """Raise ValueError unless every cached node is in the plan and may be memoised (never a
    write; a read only within its ttl_s)."""
    nodes = {node.id: node for node in wf.nodes}
    for node_id in cached:
        if node_id not in nodes:
            raise ValueError(f"cached node {node_id} is not in the plan")
        blocker = cache_blocker(nodes[node_id])
        if blocker:
            raise ValueError(f"{node_id} cannot be cached: {blocker}")


def _plan_id(wf: Workflow, order_edges: list[tuple[str, str]], cached: list[str],
             passes: tuple[str, ...]) -> str:
    """A short hash of everything the executor uses, so equal plans get equal ids."""
    payload = {"passes": list(passes), "workflow": wf.model_dump(mode="json"),
               "order_edges": [list(edge) for edge in order_edges], "cached_nodes": cached}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
