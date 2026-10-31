"""M7: adaptive pass choices come from measured profiles."""

from synapse import ROOT
from synapse.ir import NodeStats, Profile
from synapse.optimiser import optimise
from synapse.parser import load_workflow


def test_no_profile_skips_dynamic_passes() -> None:
    wf = load_workflow(ROOT / "workflows" / "loan_prescreen.yaml")
    plan = optimise(wf, None)
    assert [n.id for n in plan.workflow.nodes][-1] == "notify_customer"
    assert not plan.cached_nodes
    assert {d.pass_name for d in plan.decisions if d.status == "skipped"} == {
        "parallelise", "memoise"}


def test_measured_parallel_and_repeat_signals_select_passes() -> None:
    wf = load_workflow(ROOT / "workflows" / "loan_prescreen.yaml")
    profile = Profile(workflow=wf.name, runs=3, critical_path=["parse_app"],
                      nodes={node.id: NodeStats(node_id=node.id, p50_ms=100, p95_ms=120,
                                               tokens_in=0, tokens_out=0, cost=1,
                                               repeat_input_rate=0.5 if node.id == "bureau_check" else 0)
                             for node in wf.nodes})
    plan = optimise(wf, profile)
    selected = {d.pass_name: d for d in plan.decisions if d.status == "applied"
                and d.predicted_ms is not None}
    assert selected["parallelise"].predicted_ms > 0
    assert selected["memoise"].predicted_ms == 50
    assert selected["memoise"].predicted_cost == 0.5
    assert "bureau_check" in plan.cached_nodes
