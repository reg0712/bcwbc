"""Choose dynamic passes from measured node profiles, with explicit reasons."""

from __future__ import annotations

from synapse.compiler import compile_workflow
from synapse.graph import build_graph, critical_path
from synapse.ir import Decision, Effect, ExecPlan, Profile, Workflow
from synapse.passes.parallelise import safe_order_edges
from synapse.passes.routing import Routing


def optimise(wf: Workflow, profile: Profile | None,
             *, validated_routing: bool = False) -> ExecPlan:
    """Static rewrites always run; schedule and cache changes need measured evidence."""
    selected = ["dead_nodes", "cse", "compact_inputs"]
    extra: list[Decision] = []
    static = compile_workflow(wf, passes=tuple(selected))
    if profile is None:
        for name in ("parallelise", "memoise"):
            extra.append(Decision(pass_name=name, nodes=[], status="skipped",
                                  reason="no measured profile is available"))
    else:
        graph = build_graph(static.workflow)
        graph.add_edges_from(safe_order_edges(static.workflow))
        latencies = {id: profile.nodes[id].p50_ms for id in graph.nodes if id in profile.nodes}
        serial = sum(latencies.values())
        _, longest = critical_path(graph, latencies)
        parallel_gain = max(0.0, serial - longest)
        if parallel_gain > 0:
            selected.append("parallelise")
            extra.append(Decision(pass_name="parallelise", nodes=list(graph.nodes),
                                  status="applied", reason="profile shows independent work off the critical path",
                                  predicted_ms=parallel_gain))
        else:
            extra.append(Decision(pass_name="parallelise", nodes=[], status="skipped",
                                  reason="no measured parallel latency gain"))
        if validated_routing:
            selected.append("routing")
            candidates = [d.nodes[0] for d in Routing(probe=True).plan(static.workflow)
                          if d.status == "applied"]
            extra.append(Decision(pass_name="routing", nodes=candidates,
                                  status="applied", reason="candidate passed the validation suite"))
        else:
            extra.append(Decision(pass_name="routing", nodes=[], status="skipped",
                                  reason="routing has not passed the validation suite"))
        repeaters = [id for id in graph.nodes if id in profile.nodes
                     and profile.nodes[id].repeat_input_rate > 0
                     and (graph.nodes[id]["node"].effect is Effect.PURE or
                          (graph.nodes[id]["node"].effect is Effect.READ
                           and graph.nodes[id]["node"].ttl_s))]
        cache_gain_ms = sum(profile.nodes[id].p50_ms * profile.nodes[id].repeat_input_rate
                            for id in repeaters)
        cache_gain_cost = sum(profile.nodes[id].cost * profile.nodes[id].repeat_input_rate
                              for id in repeaters)
        if repeaters:
            selected.append("memoise")
            extra.append(Decision(pass_name="memoise", nodes=repeaters,
                                  status="applied", reason="identical inputs recur in measured runs",
                                  predicted_ms=cache_gain_ms, predicted_cost=cache_gain_cost))
        else:
            extra.append(Decision(pass_name="memoise", nodes=[], status="skipped",
                                  reason="no repeated inputs observed"))
    routing_evidence = ({node_id: None for node_id in candidates}
                        if "routing" in selected else None)
    plan = compile_workflow(wf, passes=tuple(selected), routing_evidence=routing_evidence)
    plan.decisions.extend(extra)
    return plan
