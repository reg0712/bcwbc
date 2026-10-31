"""Backends (M2): output schemas, prompt rendering, the mock LLM, tool calls and retrieval."""

from __future__ import annotations

import asyncio
import math
import time

import httpx
import pytest
from pydantic import ValidationError

from mock_services import MOCK_TASKS
from synapse import ROOT
from synapse.backends.llm import (MockLLM, extract_inputs, load_schema, output_schema,
                                  render_prompt)
from synapse.backends.retrieval import EmbeddingRetriever, KeywordRetriever
from synapse.backends.tools import ToolClient, tool_arguments
from synapse.ir import Node, NodeKind
from synapse.parser import load_workflow

LOAN = load_workflow(ROOT / "workflows" / "loan_prescreen.yaml")
NODES = {node.id: node for node in LOAN.nodes}
SEGMENT_INPUTS = {"applicant.income": 140000, "applicant.employment": "salaried"}
RISK = {"decision": "refer", "kyc_verified": True, "bureau_score": 690, "foir": 0.143,
        "reasons": ["bureau score 690 is in the 650 to 720 refer band"], "summary": "Refer."}


def mock_llm(scale: float) -> MockLLM:
    return MockLLM(MOCK_TASKS, {"small": 300, "large": 1500},
                   {"small": "mock-small", "large": "mock-large"}, scale)


def test_every_llm_node_has_a_template_with_an_inputs_slot_and_a_schema() -> None:
    for node in (n for n in LOAN.nodes if n.kind is NodeKind.LLM):
        assert "{inputs}" in (ROOT / str(node.prompt)).read_text(), node.id
        assert output_schema(node) is not None, node.id


def test_schema_models_are_strict() -> None:
    risk = load_schema("risk.json")
    assert risk.model_validate(RISK).decision == "refer"
    for bad in ({**RISK, "decision": "maybe"}, {**RISK, "extra": 1},
                {key: value for key, value in RISK.items() if key != "reasons"}):
        with pytest.raises(ValidationError):
            risk.model_validate(bad)


def test_rendered_prompt_puts_instructions_first_and_inputs_last() -> None:
    prompt = render_prompt(NODES["classify_segment"], SEGMENT_INPUTS)
    assert prompt.startswith("Classify a loan applicant")
    assert prompt.rstrip().endswith("</inputs>")
    assert extract_inputs(prompt) == SEGMENT_INPUTS


def test_mock_llm_returns_schema_valid_output_and_estimates_tokens() -> None:
    node = NODES["classify_segment"]
    prompt = render_prompt(node, SEGMENT_INPUTS)
    output, usage = asyncio.run(mock_llm(0.001).invoke(node, prompt, output_schema(node)))
    assert output == {"label": "salaried_prime"}
    assert usage.tokens_in == math.ceil(len(prompt) / 4)
    assert usage.tokens_out > 0
    assert usage.model == "mock-large"


def test_mock_llm_waits_for_its_tier_latency() -> None:
    node = NODES["classify_segment"]
    started = time.perf_counter()
    asyncio.run(mock_llm(0.05).invoke(node, render_prompt(node, SEGMENT_INPUTS), None))
    assert time.perf_counter() - started >= 1.5 * 0.05  # large tier: 1,500 ms at scale 0.05


def test_mock_llm_refuses_prompts_it_has_no_task_for() -> None:
    node = Node(id="x", kind="llm", model="small", prompt="prompts/unknown.txt", effect="pure")
    with pytest.raises(LookupError, match="unknown.txt"):
        asyncio.run(mock_llm(0.001).invoke(node, "<inputs>\n{}\n</inputs>", None))


def test_tool_arguments_are_named_by_the_last_path_segment() -> None:
    assert tool_arguments({"applicant.pan": "P", "customer_id": "C"}) == {"pan": "P",
                                                                          "customer_id": "C"}
    with pytest.raises(ValueError, match="pan"):
        tool_arguments({"applicant.pan": 1, "spouse.pan": 2})


def test_unknown_tool_raises_before_any_request() -> None:
    async def call() -> None:
        async with httpx.AsyncClient(base_url="http://unused.invalid") as client:
            await ToolClient(client).call("teleport_api", {})
    with pytest.raises(LookupError, match="teleport_api"):
        asyncio.run(call())


def test_keyword_retriever_ranks_the_eligibility_policy_first() -> None:
    hits = asyncio.run(KeywordRetriever().search("lending_policy", "personal loan eligibility"))
    assert [hit["id"] for hit in hits][0] == "lp-01"
    assert len(hits) == 3


def test_embedding_retriever_finds_the_eligibility_policy(embedding_model: str) -> None:
    hits = asyncio.run(EmbeddingRetriever(embedding_model).search(
        "lending_policy", "personal loan eligibility"))
    assert "lp-01" in [hit["id"] for hit in hits]
    assert hits == sorted(hits, key=lambda hit: -hit["score"])
