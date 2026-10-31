"""YAML workflow files to Workflow models, validated on load.

A file has `workflow` (the name), `input` (the input fields), `nodes`, and optionally
`output` (results that must survive optimization; each must be produced by some node). Unknown
keys and node fields are errors, so a typo such as `efect: read` never falls back silently to
a default. Settings that would be silently ignored are errors too: `fallback_model` outside a
checked small-model LLM node, and `ttl_s` on a write (a node without `effect` is a write).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from synapse.ir import Effect, Node, NodeKind, Workflow
from synapse.passes.cascade import checkable


class ParseError(ValueError):
    """A workflow file that does not describe a valid workflow."""


# Accepted spellings of each top-level key, preferred spelling first.
_TOP_LEVEL = {"name": ("workflow", "name"), "inputs": ("input", "inputs"),
              "nodes": ("nodes",), "outputs": ("output", "outputs")}

# Fields a node of each kind must set to be runnable.
_REQUIRED_BY_KIND = {
    NodeKind.LLM: ("model", "prompt"),
    NodeKind.TOOL: ("tool",),
    NodeKind.RETRIEVE: ("index", "query"),
}

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_REFERENCE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


def load_workflow(path: str | Path) -> Workflow:
    """Read and validate a workflow YAML file."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ParseError(f"{path}: not valid YAML: {exc}") from exc
    return parse_workflow(data, source=str(path))


def parse_workflow(data: Any, source: str = "<workflow>") -> Workflow:
    """Validate a loaded YAML document into a Workflow; errors name the source and node."""
    if not isinstance(data, dict):
        raise ParseError(f"{source}: expected a mapping with workflow, input and nodes")
    known = {key for spellings in _TOP_LEVEL.values() for key in spellings}
    unknown = sorted(str(key) for key in data if key not in known)
    if unknown:
        raise ParseError(f"{source}: unknown top-level key(s): {', '.join(unknown)}")
    name = _pick(data, _TOP_LEVEL["name"], source)
    inputs = _pick(data, _TOP_LEVEL["inputs"], source)
    nodes = _pick(data, _TOP_LEVEL["nodes"], source)
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ParseError(f"{source}: workflow name must be a plain identifier, got {name!r}")
    if not isinstance(inputs, list) or not all(
            isinstance(field, str) and _NAME.fullmatch(field) for field in inputs):
        raise ParseError(f"{source}: input must be a list of plain field names")
    if len(set(inputs)) != len(inputs):
        raise ParseError(f"{source}: input lists a field twice")
    if not isinstance(nodes, list) or not nodes:
        raise ParseError(f"{source}: nodes must be a non-empty list")
    parsed = [_parse_node(raw, position, source) for position, raw in enumerate(nodes, start=1)]
    results = _pick(data, _TOP_LEVEL["outputs"], source, default=[])
    _check_outputs(results, parsed, source)
    return Workflow(name=name, inputs=inputs, nodes=parsed, outputs=results)


_REQUIRED = object()  # _pick default: the key must be present


def _pick(data: dict[Any, Any], spellings: tuple[str, ...], source: str,
          default: Any = _REQUIRED) -> Any:
    """The value under exactly one accepted spelling of a top-level key, or default when the
    key is optional and absent."""
    present = [key for key in spellings if key in data]
    if len(present) > 1:
        raise ParseError(f"{source}: use only one of {' or '.join(present)}")
    if not present:
        if default is _REQUIRED:
            raise ParseError(f"{source}: missing {spellings[0]!r}")
        return default
    return data[present[0]]


def _check_outputs(fields: Any, nodes: list[Node], source: str) -> None:
    """Workflow outputs: field names or dotted paths, each listed once and produced by a node."""
    if not isinstance(fields, list) or not all(
            isinstance(field, str) and _REFERENCE.fullmatch(field) for field in fields):
        raise ParseError(f"{source}: output must be a list of field names or dotted paths")
    if len(set(fields)) != len(fields):
        raise ParseError(f"{source}: output lists a field twice")
    produced = {field for node in nodes for field in node.outputs}
    missing = [field for field in fields if field.split(".", 1)[0] not in produced]
    if missing:
        raise ParseError(f"{source}: output {missing[0]!r} is not produced by any node")


def _parse_node(raw: Any, position: int, source: str) -> Node:
    """Validate one node mapping; errors name the node by id, or by position without one."""
    if not isinstance(raw, dict):
        raise ParseError(f"{source}: node {position} is not a mapping")
    where = f"{source}: node {raw.get('id', position)}"
    unknown = sorted(str(key) for key in raw if key not in Node.model_fields)
    if unknown:
        raise ParseError(f"{where}: unknown field(s): {', '.join(unknown)}")
    try:
        node = Node.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"])
        raise ParseError(f"{where}: {field}: {first['msg']}") from exc
    _check_node(node, where)
    return node


def _check_node(node: Node, where: str) -> None:
    """Check names, kind-specific fields, expiry and fallback, which the IR types alone cannot."""
    if not _NAME.fullmatch(node.id):
        raise ParseError(f"{where}: id must be a plain identifier")
    bad = [field for field in node.outputs if not _NAME.fullmatch(field)]
    if bad:
        raise ParseError(f"{where}: outputs must be plain field names, got {', '.join(bad)}")
    bad = [field for field in node.inputs if not _REFERENCE.fullmatch(field)]
    if bad:
        raise ParseError(f"{where}: inputs must be field names or dotted paths, "
                         f"got {', '.join(bad)}")
    missing = [field for field in _REQUIRED_BY_KIND.get(node.kind, ()) if not getattr(node, field)]
    if missing:
        raise ParseError(f"{where}: {node.kind.value} nodes need {', '.join(missing)}")
    if node.ttl_s is not None and node.ttl_s <= 0:
        raise ParseError(f"{where}: ttl_s must be positive")
    if node.ttl_s is not None and node.effect is Effect.WRITE:
        raise ParseError(f"{where}: ttl_s is a read's cache expiry, but this node is a write "
                         f"(a node without effect counts as a write)")
    if node.fallback_model is not None:
        if node.kind is not NodeKind.LLM:
            raise ParseError(f"{where}: fallback_model applies only to llm nodes")
        if node.model != "small":
            raise ParseError(f"{where}: fallback_model needs model: small, the tier it "
                             f"escalates from")
        if not checkable(node):
            raise ParseError(f"{where}: fallback_model needs a schema test to decide when "
                             f"to escalate")
