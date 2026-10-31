"""Model-backed LangGraph research: two local searches, one grounded synthesis call.

NODE_SPECS makes import static: no model or service is called to discover I/O.
Native LangGraph defaults to mock; build_graph(adapter) accepts a live LLMAdapter.
SYNAPSE's executor uses the model selected in the CLI/dashboard. Dataset evidence
is local; citation/schema checks do not establish HotpotQA answer accuracy.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from synapse.backends import mock_backends
from synapse.backends.llm import output_schema, render_prompt
from synapse.ir import Node
from workflows.research_agent_langgraph import SAMPLE_INPUTS, _WORDS

WORKFLOW_NAME = "research_llm"
OUTPUTS = ["answer"]


class State(TypedDict, total=False):
    question: str
    context: dict
    evidence_one: dict
    evidence_two: dict
    answer: dict


def _search(state: State, rank: int) -> dict:
    words = set(_WORDS.findall(state["question"].lower()))
    passages = [{"title": title, "text": " ".join(sentences)} for title, sentences in
                zip(state["context"]["title"], state["context"]["sentences"], strict=True)]
    ranked = sorted(passages, key=lambda p: -len(words & set(_WORDS.findall(
        f"{p['title']} {p['text']}".lower()))))
    return ranked[min(rank, len(ranked) - 1)]


def search_one(state: State) -> dict:
    return {"evidence_one": _search(state, 0)}


def search_two(state: State) -> dict:
    return {"evidence_two": _search(state, 1)}


async def synthesise(state: State, adapter=None) -> dict:
    node = Node(id="synthesise", **NODE_SPECS["synthesise"])
    inputs = {key: state[key] for key in node.inputs}
    if adapter is None:
        async with mock_backends(scale=0) as backends:
            return await synthesise(state, backends.llm)
    result, _ = await adapter.invoke(node, render_prompt(node, inputs), output_schema(node))
    return {"answer": result}


NODES = {"search_one": search_one, "search_two": search_two, "synthesise": synthesise}
EFFECTS = {name: "pure" for name in NODES}
NODE_SPECS = {
    "search_one": {"kind": "transform", "tool": "search_one", "effect": "pure",
                   "deterministic": True, "inputs": ["question", "context"], "outputs": ["evidence_one"]},
    "search_two": {"kind": "transform", "tool": "search_two", "effect": "pure",
                   "deterministic": True, "inputs": ["question", "context"], "outputs": ["evidence_two"]},
    "synthesise": {"kind": "llm", "model": "large", "effect": "pure",
                   "prompt": "prompts/research_answer.txt", "max_output_tokens": 768,
                   "inputs": ["question", "evidence_one", "evidence_two"], "outputs": ["answer"],
                   "tests": [{"schema": "research_answer.json"}]},
}


def build_graph(adapter=None) -> StateGraph:
    """Two independent searches feed one synthesis, avoiding two intermediate LLM summaries."""
    graph = StateGraph(State)
    graph.add_node("search_one", search_one)
    graph.add_node("search_two", search_two)

    async def answer_node(state: State) -> dict:
        return await synthesise(state, adapter)

    graph.add_node("synthesise", answer_node)
    graph.add_edge(START, "search_one")
    graph.add_edge("search_one", "search_two")
    graph.add_edge("search_two", "synthesise")
    graph.add_edge("synthesise", END)
    return graph
