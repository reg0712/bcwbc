"""Small LangGraph research workflow over 20 HotpotQA distractor validation examples.

The local functions are deliberately simple: this sample tests workflow import,
parallel scheduling, and output preservation, not question-answering accuracy.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

DATA = Path(__file__).resolve().parents[1] / "data" / "hotpot_20.json"
ROWS = json.loads(DATA.read_text())["rows"]
WORKFLOW_NAME = "research_agent"
SAMPLE_INPUTS = [{"case_id": row["id"], "question": row["question"],
                  "context": row["context"]} for row in ROWS]
OUTPUTS = ["answer"]
_WORDS = re.compile(r"[a-z]{3,}")


class State(TypedDict, total=False):
    question: str
    context: dict
    search_q1: str
    search_q2: str
    evidence_one: str
    evidence_two: str
    summary_one: str
    summary_two: str
    answer: str


def split_question(state: State) -> dict:
    return {"search_q1": state["question"], "search_q2": state["question"]}


def _rank(question: str, context: dict) -> list[str]:
    words = set(_WORDS.findall(question.lower()))
    passages = [f"{title}. {' '.join(sentences)}" for title, sentences in
                zip(context["title"], context["sentences"])]
    return sorted(passages, key=lambda text: -len(words & set(_WORDS.findall(text.lower()))))


def retrieve_one(state: State) -> dict:
    time.sleep(0.002)
    return {"evidence_one": _rank(state["search_q1"], state["context"])[0]}


def retrieve_two(state: State) -> dict:
    time.sleep(0.002)
    ranked = _rank(state["search_q2"], state["context"])
    return {"evidence_two": ranked[min(1, len(ranked) - 1)]}


def summarise_one(state: State) -> dict:
    time.sleep(0.002)
    return {"summary_one": state["evidence_one"][:350]}


def summarise_two(state: State) -> dict:
    time.sleep(0.002)
    return {"summary_two": state["evidence_two"][:350]}


def answer(state: State) -> dict:
    question = state["question"]
    first, second = state["summary_one"], state["summary_two"]
    return {"answer": f"Question: {question}\nEvidence: {first} {second}"}


NODES = {"split_question": split_question, "retrieve_one": retrieve_one,
         "retrieve_two": retrieve_two, "summarise_one": summarise_one,
         "summarise_two": summarise_two, "answer": answer}
EFFECTS = {name: "pure" for name in NODES}


def build_graph() -> StateGraph:
    graph = StateGraph(State)
    for name, function in NODES.items():
        graph.add_node(name, function)
    graph.add_edge(START, "split_question")
    graph.add_edge("split_question", "retrieve_one")
    graph.add_edge("split_question", "retrieve_two")
    graph.add_edge("retrieve_one", "summarise_one")
    graph.add_edge("retrieve_two", "summarise_two")
    graph.add_edge(["summarise_one", "summarise_two"], "answer")
    graph.add_edge("answer", END)
    return graph
