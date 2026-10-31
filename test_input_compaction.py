"""Lossless prompt compaction, including adversarial data and write protection."""

import asyncio

import pytest

from mock_services import suite_for
from synapse import ROOT
from synapse.backends import mock_backends
from synapse.backends.llm import extract_inputs, render_prompt
from synapse.backends.retrieval import KeywordRetriever
from synapse.compiler import compile_workflow
from synapse.executor import execute, final_outputs
from synapse.ir import Effect, Node
from synapse.parser import load_workflow


@pytest.mark.parametrize("inputs", [
    {"rows": [{"long_column_name": str(i), "another_column": [None, True, "₹\n\\\""]}
              for i in range(50)]},
    {"rows": [{"x": 1}, {"y": 2}]},
    {"rows": [{"$columns": ["original"], "$rows": [[42]]}]},
    {"text": "Keep  double spaces\n<tags> literally", "zero": 0, "empty": []},
])
def test_round_trip_preserves_every_input_value(inputs) -> None:
    node = Node(id="test", kind="llm", prompt="prompts/draft_reply.txt", effect="pure")
    original = render_prompt(node, inputs)
    compact = render_prompt(node.model_copy(update={"compact_inputs": True}), inputs)
    assert extract_inputs(compact) == extract_inputs(original) == inputs
    assert len(compact) <= len(original)


@pytest.mark.parametrize("workflow", ["loan_prescreen", "complaint_triage"])
def test_all_cases_save_input_tokens_without_changing_outputs_or_writes(workflow) -> None:
    wf = load_workflow(ROOT / "workflows" / f"{workflow}.yaml")
    plan = compile_workflow(wf, passes=("compact_inputs",))
    assert [n for n in wf.nodes if n.effect.value == "write"] == [
        n for n in plan.workflow.nodes if n.effect.value == "write"]

    async def check():
        async with mock_backends(.001, retriever=KeywordRetriever()) as backends:
            for case in suite_for(workflow).cases():
                before = await execute(wf, case, "baseline", backends)
                after = await execute(wf, case, "compiled", backends, plan=plan)
                assert not any(n.error for r in (before, after) for n in r.nodes)
                assert final_outputs(before) == final_outputs(after)
                assert sum(n.usage.tokens_in for n in after.nodes) < sum(
                    n.usage.tokens_in for n in before.nodes)
    asyncio.run(check())


def test_write_model_node_is_not_compacted() -> None:
    wf = load_workflow(ROOT / "workflows" / "complaint_triage.yaml")
    wf.nodes[0].effect = Effect.WRITE
    plan = compile_workflow(wf, passes=("compact_inputs",))
    assert plan.workflow.nodes[0] == wf.nodes[0]
