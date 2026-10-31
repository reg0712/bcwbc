"""YAML parser (M1): accepted spellings, strict fields, kind requirements and errors that name
the file and node."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from synapse.ir import Effect, NodeKind
from synapse.parser import ParseError, load_workflow, parse_workflow

TOOL = {"id": "t", "kind": "tool", "tool": "api", "inputs": ["x"], "outputs": ["y"],
        "effect": "read"}
LLM = {"id": "l", "kind": "llm", "model": "large", "prompt": "prompts/l.txt", "inputs": ["x"],
       "outputs": ["z"], "effect": "pure"}
RETRIEVE = {"id": "r", "kind": "retrieve", "index": "policy", "query": "eligibility",
            "outputs": ["p"], "effect": "pure"}


def doc(nodes: list[Any] | None = None, **top: Any) -> dict[str, Any]:
    """A minimal workflow document; keyword arguments add or replace top-level keys."""
    return {"workflow": "toy", "input": ["x"],
            "nodes": nodes if nodes is not None else [TOOL], **top}


def tool(drop: tuple[str, ...] = (), **fields: Any) -> dict[str, Any]:
    """The base tool node with fields replaced or added and the named fields removed."""
    raw = {**TOOL, **fields}
    for key in drop:
        del raw[key]
    return raw


def test_minimal_workflow_parses() -> None:
    wf = parse_workflow(doc([TOOL, LLM, RETRIEVE]))
    assert wf.name == "toy"
    assert [(n.id, n.kind, n.effect) for n in wf.nodes] == [
        ("t", NodeKind.TOOL, Effect.READ), ("l", NodeKind.LLM, Effect.PURE),
        ("r", NodeKind.RETRIEVE, Effect.PURE)]


def test_name_and_inputs_spellings_are_accepted() -> None:
    wf = parse_workflow({"name": "toy", "inputs": ["x"], "nodes": [TOOL]})
    assert (wf.name, wf.inputs) == ("toy", ["x"])


def test_missing_effect_means_write() -> None:
    assert parse_workflow(doc([tool(drop=("effect",))])).nodes[0].effect is Effect.WRITE


@pytest.mark.parametrize(("data", "message"), [
    pytest.param([], "expected a mapping", id="not-a-mapping"),
    pytest.param(doc(extra=1), "unknown top-level key", id="unknown-top-level-key"),
    pytest.param(doc(name="other"), "only one of workflow or name", id="both-spellings"),
    pytest.param({"workflow": "toy", "input": ["x"]}, "missing 'nodes'", id="no-nodes"),
    pytest.param(doc([]), "non-empty list", id="empty-nodes"),
    pytest.param(doc(workflow="loan prescreen"), "plain identifier", id="bad-name"),
    pytest.param(doc(input="x"), "list of plain field names", id="input-not-a-list"),
    pytest.param(doc(input=["x", "x"]), "twice", id="duplicate-input"),
    pytest.param(doc(["oops"]), "node 1 is not a mapping", id="node-not-a-mapping"),
    pytest.param(doc([tool(efect="read")]), "unknown field.*efect", id="typo-field"),
    pytest.param(doc([tool(kind="agent")]), "kind", id="unknown-kind"),
    pytest.param(doc([{**LLM, "model": "medium"}]), "model", id="unknown-model-tier"),
    pytest.param(doc([{**LLM, "prompt": None}]), "llm nodes need prompt", id="llm-no-prompt"),
    pytest.param(doc([tool(drop=("tool",))]), "tool nodes need tool", id="tool-no-tool"),
    pytest.param(doc([{**RETRIEVE, "query": ""}]), "retrieve nodes need query",
                 id="retrieve-no-query"),
    pytest.param(doc([tool(id="bad id")]), "id must be a plain identifier", id="bad-id"),
    pytest.param(doc([tool(outputs=["y.z"])]), "outputs must be plain field names",
                 id="dotted-output"),
    pytest.param(doc([tool(inputs=["x..y"])]), "inputs must be field names or dotted paths",
                 id="bad-input-path"),
    pytest.param(doc([tool(ttl_s=0)]), "ttl_s must be positive", id="zero-ttl"),
])
def test_invalid_workflows_raise(data: Any, message: str) -> None:
    with pytest.raises(ParseError, match=message):
        parse_workflow(data)


def test_errors_name_the_file_and_node(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("workflow: toy\ninput: [x]\nnodes:\n"
                    "  - {id: fetch, kind: tool, inputs: [x], outputs: [y]}\n")
    with pytest.raises(ParseError, match=r"bad\.yaml: node fetch: tool nodes need tool"):
        load_workflow(path)


def test_invalid_yaml_raises_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("nodes: [unclosed\n")
    with pytest.raises(ParseError, match="not valid YAML"):
        load_workflow(path)
