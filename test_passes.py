"""Optimisation passes one at a time: each legality rule of the spec table, the Decision
records (applied, skipped and rejected, each with a reason), and apply() refusing unsafe
decisions handed to it."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from synapse.backends import open_backends
from synapse.backends.retrieval import KeywordRetriever
from synapse.compiler import compile_workflow
from synapse.executor import execute
from synapse.ir import Decision, Node, Workflow
from synapse.passes.cse import CSE
from synapse.passes.dead_nodes import DeadNodes
from synapse.passes.memoise import ExactCache, Memoise, cache_key
from synapse.passes.parallelise import Parallelise, parallel_levels, safe_order_edges
from synapse.passes.routing import Routing, final_steps

SCHEMA = [{"schema": "segment.json"}]


def node(id: str, **fields: Any) -> Node:
    """A pure, deterministic transform reading x unless fields say otherwise."""
    return Node(**{"id": id, "kind": "transform", "effect": "pure", "deterministic": True,
                   "inputs": ["x"], "outputs": [f"{id}_out"], **fields})


def retrieve(id: str, **fields: Any) -> Node:
    return node(id, **{"kind": "retrieve", "index": "lending_policy", "query": "eligibility",
                       "inputs": [], **fields})


def tool(id: str, **fields: Any) -> Node:
    """A deterministic read tool with a one-minute expiry."""
    return node(id, **{"kind": "tool", "tool": "bureau_api", "effect": "read", "ttl_s": 60,
                       **fields})


def llm(id: str, **fields: Any) -> Node:
    """A pure large-model call with a schema test (not marked deterministic: temperature 0)."""
    return node(id, **{"kind": "llm", "model": "large", "prompt": "prompts/classify_segment.txt",
                       "deterministic": False, "tests": SCHEMA, **fields})


def flow(*nodes: Node, outputs: list[str] | None = None) -> Workflow:
    """A workflow with input x; by default every node's outputs are workflow outputs."""
    if outputs is None:
        outputs = [field for n in nodes for field in n.outputs]
    return Workflow(name="toy", inputs=["x"], nodes=list(nodes), outputs=outputs)


def by_status(decisions: list[Decision]) -> dict[str, list[list[str]]]:
    grouped: dict[str, list[list[str]]] = {}
    for decision in decisions:
        grouped.setdefault(decision.status, []).append(decision.nodes)
    return grouped


def applied(decision: Decision) -> Decision:
    return decision.model_copy(update={"status": "applied"})


# Dead-node elimination


def test_dead_nodes_reach_the_fixed_point_through_chains() -> None:
    wf = flow(tool("fetch"), node("shape", inputs=["fetch_out"]),
              node("summarise", inputs=["shape_out.text"]), node("answer"),
              outputs=["answer_out"])
    decisions = DeadNodes().plan(wf)
    assert by_status(decisions) == {"applied": [["fetch"], ["shape"], ["summarise"]]}
    assert "fetch removed: nothing that stays reads fetch_out" in decisions[0].reason
    plan = compile_workflow(wf, passes=("dead_nodes",))
    assert [n.id for n in plan.workflow.nodes] == ["answer"]


def test_dead_nodes_keep_writes_required_checks_and_output_producers() -> None:
    untagged = Node(id="crm", kind="tool", tool="crm_api", inputs=["x"], outputs=["crm_out"])
    wf = flow(node("sms", kind="tool", tool="sms_gateway", effect="write"), untagged,
              tool("audit", required=True), tool("feeds_required"),
              tool("needs", inputs=["feeds_required_out"], required=True),
              node("risk"), node("unused"),
              outputs=["risk_out.decision"])
    decisions = DeadNodes().plan(wf)
    assert by_status(decisions) == {"skipped": [["sms"], ["crm"], ["audit"], ["needs"]],
                                    "applied": [["unused"]]}
    reasons = {d.nodes[0]: d.reason for d in decisions}
    assert reasons["sms"] == "sms kept although nothing reads its result: write effect"
    assert reasons["crm"].endswith("write effect")
    assert reasons["audit"].endswith("required check")
    kept = [n.id for n in compile_workflow(wf, passes=("dead_nodes",)).workflow.nodes]
    assert kept == ["sms", "crm", "audit", "feeds_required", "needs", "risk"]


@pytest.mark.parametrize(("removed", "message"), [
    pytest.param("sms", "write effect", id="write"),
    pytest.param("check", "required check", id="required"),
    pytest.param("risk", "produces workflow output risk_out", id="workflow-output"),
    pytest.param("fetch", "risk still reads it", id="still-read"),
])
def test_dead_node_apply_refuses_unsafe_removals(removed: str, message: str) -> None:
    wf = flow(node("sms", kind="tool", tool="sms_gateway", effect="write"),
              tool("check", required=True), tool("fetch"), node("risk", inputs=["fetch_out"]),
              outputs=["risk_out"])
    forged = Decision(pass_name="dead_nodes", nodes=[removed], status="applied", reason="forged")
    with pytest.raises(ValueError, match=message):
        DeadNodes().apply(wf, [forged])


# Common-subexpression elimination


def test_cse_merges_duplicates_of_duplicates_and_rewires_dotted_inputs() -> None:
    wf = flow(retrieve("policy_1"), retrieve("policy_2"),
              llm("judge_1", inputs=["policy_1_out.text"]),
              llm("judge_2", inputs=["policy_2_out.text"]),
              llm("final", inputs=["judge_1_out", "judge_2_out.label", "x"]),
              outputs=["final_out"])
    decisions = CSE().plan(wf)
    assert by_status(decisions) == {"applied": [["policy_1", "policy_2"], ["judge_1", "judge_2"]]}
    assert decisions[0].reason == ("policy_2 merged into policy_1: identical retrieve step "
                                   "with the same inputs")
    merged = compile_workflow(wf, passes=("cse",)).workflow
    assert [n.id for n in merged.nodes] == ["policy_1", "judge_1", "final"]
    assert merged.nodes[2].inputs == ["judge_1_out", "judge_1_out.label", "x"]


@pytest.mark.parametrize(("change", "value"), [
    ("kind", "tool"), ("effect", "read"), ("inputs", ["y", "x"]), ("model", "small"),
    ("prompt", "prompts/risk_summary.txt"), ("tool", "other_api"), ("index", "other_index"),
    ("query", "other question"), ("ttl_s", 5), ("tests", [{"schema": "risk.json"}]),
])
def test_cse_merges_only_exact_copies(change: str, value: Any) -> None:
    common = {"kind": "llm", "effect": "pure", "model": "large",
              "prompt": "prompts/classify_segment.txt", "tool": "api", "index": "lending_policy",
              "query": "q", "tests": SCHEMA, "inputs": ["x", "y"]}
    first = Node(id="a", outputs=["a_out"], **common)
    second = Node(id="b", outputs=["b_out"], **{**common, change: value})
    wf = Workflow(name="toy", inputs=["x", "y"], nodes=[first, second],
                  outputs=["a_out", "b_out"])
    assert CSE().plan(wf) == []


def test_cse_never_merges_writes_untagged_nodes_or_nondeterministic_tools() -> None:
    sms = {"kind": "tool", "tool": "sms_gateway", "effect": "write"}
    wf = flow(node("sms_1", **sms), node("sms_2", **sms),
              Node(id="crm_1", kind="tool", tool="crm", inputs=["x"], outputs=["crm_1_out"]),
              Node(id="crm_2", kind="tool", tool="crm", inputs=["x"], outputs=["crm_2_out"]),
              tool("otp_1", tool="otp_api", deterministic=False),
              tool("otp_2", tool="otp_api", deterministic=False),
              tool("audit_1", required=True), tool("audit_2", required=True),
              llm("label_1", inputs=["x"]), llm("label_2", inputs=["x"]),
              outputs=["sms_1_out", "label_2_out.label"])
    reasons = {d.nodes[1]: (d.status, d.reason) for d in CSE().plan(wf)}
    assert reasons["sms_2"] == ("skipped", "sms_2 not merged into sms_1: write effect")
    assert reasons["crm_2"] == ("skipped", "crm_2 not merged into crm_1: write effect")
    assert reasons["otp_2"] == ("skipped", "otp_2 not merged into otp_1: not deterministic")
    assert reasons["audit_2"] == ("skipped", "audit_2 not merged into audit_1: required check")
    # label_2 produces a workflow output, so it stays; LLM calls count as deterministic
    assert reasons["label_2"][0] == "skipped" and "workflow output label_2_out" in reasons["label_2"][1]


def test_cse_merges_llm_calls_at_temperature_zero() -> None:
    wf = flow(llm("label_1"), llm("label_2"), llm("use", inputs=["label_1_out", "label_2_out"]),
              outputs=["use_out"])
    assert by_status(CSE().plan(wf)) == {"applied": [["label_1", "label_2"]]}


def test_cse_keeps_transforms_apart_because_they_pick_outputs_by_name() -> None:
    wf = flow(node("f"), node("g"))  # same inputs, but different functions (looked up by id)
    plan = compile_workflow(wf, passes=("cse",))
    assert [n.id for n in plan.workflow.nodes] == ["f", "g"]
    assert plan.workflow.outputs == ["f_out", "g_out"]


def test_cse_keeps_tool_argument_names() -> None:
    whole = flow(tool("bureau_1"), tool("bureau_2"),
                 tool("decide", tool="decide_api", inputs=["bureau_1_out", "bureau_2_out"]),
                 outputs=["decide_out"])
    (decision,) = CSE().plan(whole)
    assert decision.status == "skipped"
    assert decision.reason.endswith("decide would be sent bureau_2_out under another name")
    dotted = flow(tool("bureau_1"), tool("bureau_2"),
                  tool("decide", tool="decide_api", inputs=["bureau_1_out.pan", "bureau_2_out.score"]),
                  outputs=["decide_out"])
    plan = compile_workflow(dotted, passes=("cse",))
    assert plan.workflow.nodes[-1].inputs == ["bureau_1_out.pan", "bureau_1_out.score"]


def test_cse_does_not_merge_reads_across_a_write_but_merges_after_it() -> None:
    wf = flow(tool("balance_1"), node("transfer", kind="tool", tool="pay_api", effect="write"),
              tool("balance_2"), tool("balance_3"),
              llm("report", inputs=["balance_1_out", "balance_2_out", "balance_3_out"]),
              outputs=["report_out"])
    decisions = CSE().plan(wf)
    assert [(d.nodes, d.status) for d in decisions] == [
        (["balance_1", "balance_2"], "skipped"), (["balance_2", "balance_3"], "applied")]
    assert decisions[0].reason.endswith("write transfer runs between the two reads")


@pytest.mark.parametrize(("nodes", "message"), [
    pytest.param(["policy_1", "other"], "not identical", id="not-identical"),
    pytest.param(["policy_2", "policy_1"], "earlier node that stays", id="into-a-later-node"),
    pytest.param(["policy_1", "ghost"], "unknown node ghost", id="unknown"),
])
def test_cse_apply_rechecks_every_merge(nodes: list[str], message: str) -> None:
    wf = flow(retrieve("policy_1"), retrieve("policy_2"), retrieve("other", query="other"),
              llm("use", inputs=["policy_1_out", "policy_2_out", "other_out"]),
              outputs=["use_out"])
    forged = Decision(pass_name="cse", nodes=nodes, status="applied", reason="forged")
    with pytest.raises(ValueError, match=message):
        CSE().apply(wf, [forged])


# Parallelisation and write barriers


ORDERED = flow(node("p1"), tool("r1"), node("w1", kind="tool", tool="crm", effect="write"),
               node("p2"), tool("r2"), node("w2", kind="tool", tool="crm", effect="write"),
               outputs=["w2_out"])


def test_writes_keep_their_order_and_wait_for_every_earlier_step() -> None:
    edges = set(safe_order_edges(ORDERED))
    assert {("p1", "w1"), ("r1", "w1"), ("w1", "w2"), ("p2", "w2"), ("r2", "w2")} <= edges


def test_reads_after_a_write_wait_for_it_but_pure_steps_do_not() -> None:
    edges = set(safe_order_edges(ORDERED))
    assert ("w1", "r2") in edges  # read-your-writes
    assert ("w1", "p2") not in edges  # no outside contact, so free to overlap the write
    assert parallel_levels(ORDERED) == [["p1", "r1", "p2"], ["w1"], ["r2"], ["w2"]]


def test_parallelise_records_groups_writes_and_the_no_overlap_case() -> None:
    decisions = Parallelise().plan(ORDERED)
    assert [(d.nodes, d.status) for d in decisions] == [
        (["p1", "r1", "p2"], "applied"), (["w1"], "skipped"), (["w2"], "skipped")]
    assert decisions[1].reason.startswith("w1 not moved: a write waits for every step")
    chain = flow(node("a"), node("b", inputs=["a_out"]), outputs=["b_out"])
    (only,) = Parallelise().plan(chain)
    assert (only.status, only.reason) == ("skipped", "every step depends on the one before it")


def run_compiled(wf: Workflow, functions: dict[str, Any], passes: tuple[str, ...],
                 cache: ExactCache | None = None, runs: int = 1) -> list[Any]:
    """Compile wf and run it `runs` times on mock backends with the given functions."""
    plan = compile_workflow(wf, passes=passes)

    async def go() -> list[Any]:
        async with open_backends(live=False, scale=0.001, retriever=KeywordRetriever(),
                                 functions=functions) as backends:
            return [await execute(wf, {"case_id": "T", "x": 1}, "compiled", backends,
                                  plan=plan, cache=cache) for _ in range(runs)]
    return asyncio.run(go())


def test_compiled_run_reads_its_own_writes() -> None:
    log: list[str] = []

    async def write(inputs: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        log.append("written")
        return {"w_out": True}

    async def slow_pure(inputs: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.02)
        return {"p_out": 1}

    wf = flow(node("w", effect="write"), node("p"), node("r", effect="read", deterministic=True))
    (record,) = run_compiled(wf, {"w": write, "p": slow_pure, "r": lambda _: {"r_out": len(log)}},
                             ("parallelise",))
    runs = {n.node_id: n for n in record.nodes}
    assert all(n.error is None for n in record.nodes)
    assert runs["p"].start_s < runs["w"].end_s  # a pure step overlaps the write
    assert runs["r"].start_s >= runs["w"].end_s and runs["r"].output == {"r_out": 1}


# Exact memoisation


def test_cache_key_covers_everything_that_decides_the_output(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "prompts").mkdir()
    prompt, schema = tmp_path / "prompts" / "p.txt", tmp_path / "prompts" / "s.json"
    prompt.write_text("Classify {inputs}")
    schema.write_text('{"type": "object"}')
    monkeypatch.setattr("synapse.passes.memoise.ROOT", tmp_path)
    monkeypatch.setattr("synapse.backends.llm.PROMPTS", tmp_path / "prompts")
    base = Node(id="n", kind="llm", model="large", prompt="prompts/p.txt", effect="pure",
                inputs=["a", "b"], outputs=["out"], tests=[{"schema": "s.json"}])
    key = cache_key(base, {"a": 1, "b": {"c": 2}})
    assert key == cache_key(base.model_copy(), {"b": {"c": 2}, "a": 1})
    variants = [base.model_copy(update={field: value}) for field, value in [
        ("id", "m"), ("outputs", ["other"]), ("model", "small"), ("fallback_model", "large"),
        ("inputs", ["b", "a"]), ("tool", "t"), ("index", "i"), ("query", "q"),
        ("tests", [{"schema": "other.json"}])]]
    keys = {cache_key(variant, {"a": 1, "b": {"c": 2}}) for variant in variants}
    keys |= {cache_key(base, {"a": 2, "b": {"c": 2}}),
             cache_key(base, {"a": 1, "b": {"c": 2}}, settings={"provider": "gemini"})}
    assert key not in keys and len(keys) == len(variants) + 2
    prompt.write_text("Classify carefully {inputs}")
    edited_prompt = cache_key(base, {"a": 1, "b": {"c": 2}})
    schema.write_text('{"type": "object", "required": ["label"]}')
    edited_schema = cache_key(base, {"a": 1, "b": {"c": 2}})
    assert len({key, edited_prompt, edited_schema}) == 3


def test_a_write_drops_reads_of_its_entity_and_reads_of_unknown_entities() -> None:
    cache = ExactCache()
    cache.put("pure", {"v": 0}, None, None)
    cache.put("c1", {"v": 1}, 60, "C1")
    cache.put("c2", {"v": 2}, 60, "C2")
    cache.put("unknown", {"v": 3}, 60, None)
    cache.invalidate("C1")
    assert [k for k in ("pure", "c1", "c2", "unknown") if cache.get(k)] == ["pure", "c2"]
    cache.put("unknown", {"v": 3}, 60, None)
    cache.invalidate(None)  # the write's entity is unknown, so any read may be stale
    assert [k for k in ("pure", "c2", "unknown") if cache.get(k)] == ["pure"]


def test_cache_refuses_a_read_without_expiry() -> None:
    with pytest.raises(ValueError, match="ttl_s"):
        ExactCache().put("k", {"v": 1}, None, "C1", read=True)


def test_memoise_skips_writes_unexpiring_reads_and_nondeterministic_steps() -> None:
    wf = flow(node("notify", kind="tool", tool="sms_gateway", effect="write"),
              tool("live_balance", ttl_s=None), tool("otp", deterministic=False),
              tool("bureau"), node("score"), llm("label"))
    decisions = {d.nodes[0]: (d.status, d.reason) for d in Memoise().plan(wf)}
    assert decisions["notify"] == ("skipped", "notify not cached: write effect")
    assert decisions["live_balance"][1].startswith("live_balance not cached: read without ttl_s")
    assert decisions["otp"] == ("skipped", "otp not cached: not deterministic")
    assert [n for n, (status, _) in decisions.items() if status == "applied"] == [
        "bureau", "score", "label"]
    assert compile_workflow(wf, passes=("memoise",)).cached_nodes == ["bureau", "score", "label"]
    with pytest.raises(ValueError, match="cannot cache notify: write effect"):
        Memoise().apply(wf, [Decision(pass_name="memoise", nodes=["notify"], status="applied",
                                      reason="forged")])


def test_cached_read_is_not_reused_after_a_write_with_no_known_entity() -> None:
    balance = [100]

    def pay(inputs: dict[str, Any]) -> dict[str, Any]:
        balance[0] -= 30
        return {"pay_out": True}

    wf = flow(node("read", effect="read", ttl_s=60), node("pay", effect="write"))
    cache = ExactCache()
    first, second = run_compiled(wf, {"read": lambda _: {"read_out": balance[0]}, "pay": pay},
                                 ("memoise",), cache=cache, runs=2)
    reads = [next(n for n in record.nodes if n.node_id == "read") for record in (first, second)]
    assert [(n.output["read_out"], n.cache_hit) for n in reads] == [(100, False), (70, False)]


# Model routing and the cascade


def test_final_decision_steps_are_found_from_outputs_and_writes() -> None:
    wf = flow(llm("extract"), tool("lookup", inputs=["extract_out.pan"]),
              llm("decide", inputs=["lookup_out"]), node("format", inputs=["decide_out.label"]),
              node("send", kind="tool", tool="sms_gateway", effect="write", inputs=["format_out"]),
              llm("summary", inputs=["x"]), llm("dead_end", inputs=["x"]),
              outputs=["summary_out", "send_out"])
    assert final_steps(wf) == {"decide": "write send", "summary": "workflow output summary_out"}
    undeclared = wf.model_copy(update={"outputs": []})
    assert final_steps(undeclared) == {
        "decide": "write send", "summary": "a final result (the workflow declares no outputs)",
        "dead_end": "a final result (the workflow declares no outputs)"}


def test_routing_follows_structure_not_names() -> None:
    wf = flow(llm("classify_final", outputs=["verdict"]), llm("label_customer"),
              llm("free_text", tests=[]), llm("already_small", model="small"),
              llm("uses", inputs=["label_customer_out", "free_text_out", "already_small_out"],
                  outputs=["verdict_2"]),
              outputs=["verdict", "verdict_2"])
    decisions = {d.nodes[0]: (d.status, d.reason) for d in Routing(probe=True).plan(wf)}
    assert decisions["classify_final"] == (
        "skipped", "classify_final not routed: final decision step: its result reaches "
                   "workflow output verdict")
    assert decisions["label_customer"][0] == "applied"
    assert decisions["free_text"] == (
        "skipped", "free_text not routed: no schema test to catch a wrong small-model answer")
    assert decisions["already_small"][1].endswith("already on the small model")
    routed = compile_workflow(wf, passes=("routing",), routing_probe=True).workflow
    assert [(n.id, n.model, n.fallback_model) for n in routed.nodes if n.model == "small"] == [
        ("label_customer", "small", "large"), ("already_small", "small", None)]


def test_routing_uses_small_model_evidence_when_given() -> None:
    wf = flow(llm("a"), llm("b"), llm("c"), llm("final", inputs=["a_out", "b_out", "c_out"]),
              outputs=["final_out"])
    evidence = {"a": None, "b": "case A07: label differs from the baseline"}
    plan = compile_workflow(wf, passes=("routing",), routing_evidence=evidence)
    decisions = {d.nodes[0]: (d.status, d.reason) for d in plan.decisions}
    assert decisions["a"][0] == "applied" and "passed on every test case" in decisions["a"][1]
    assert decisions["b"] == ("rejected", "b not routed: case A07: label differs from the baseline")
    assert decisions["c"] == ("skipped", "c not routed: no small-model results on the test set yet")
    assert [n.id for n in plan.workflow.nodes if n.model == "small"] == ["a"]


def test_routing_without_evidence_is_not_applied_by_an_ordinary_compile() -> None:
    wf = flow(llm("classify"), llm("final", inputs=["classify_out"]),
              outputs=["final_out"])
    plan = compile_workflow(wf, passes=("routing",))
    assert next(n for n in plan.workflow.nodes if n.id == "classify").model == "large"
    assert any(d.nodes == ["classify"] and d.status == "skipped"
               and "no small-model results" in d.reason for d in plan.decisions)
    trial = compile_workflow(wf, passes=("routing",), routing_probe=True)
    assert next(n for n in trial.workflow.nodes if n.id == "classify").model == "small"


def test_routing_apply_refuses_the_final_decision_step() -> None:
    wf = flow(llm("decide"), outputs=["decide_out"])
    forged = Decision(pass_name="routing", nodes=["decide"], status="applied", reason="forged")
    with pytest.raises(ValueError, match="cannot route decide: final decision step"):
        Routing().apply(wf, [forged])
