"""Exact memoisation: reuse a node's result when the same node sees the same inputs again.

Legal for pure nodes, and for reads within their ttl_s that no later write has invalidated;
never for writes (a node without an effect tag counts as one). A non-LLM node must also be
deterministic; LLM calls run at temperature 0. The key hashes everything that decides the
output, so a changed prompt file, schema, model tier or input value never returns a stale
result.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from synapse import ROOT
from synapse.ir import Decision, Effect, Node, NodeKind, Profile, Workflow


def _digest(path: Path) -> str | None:
    """SHA-256 of a file's bytes, or None when there is no such file."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _file_digests(node: Node) -> dict[str, str | None]:
    """Digests of the files that shape an LLM node's output: its prompt template and the
    JSON schemas its tests name, located the way the LLM adapter loads them."""
    from synapse.backends.llm import PROMPTS  # deferred: compiling needs no backends

    files: dict[str, str | None] = {}
    if node.prompt:
        files[node.prompt] = _digest(ROOT / node.prompt)
    for test in node.tests:
        if "schema" in test:
            files[f"schema:{test['schema']}"] = _digest(PROMPTS / str(test["schema"]))
    return files


def cache_key(node: Node, inputs: Mapping[str, Any],
              settings: Mapping[str, Any] | None = None) -> str:
    """Hash of everything that decides a node's output: the whole node (id, kind, inputs and
    outputs, model tier and fallback, prompt, tool, index, query, expiry, tests), the contents
    of its prompt and schema files, the input values, temperature 0, and settings such as the
    provider that serves the model tier."""
    payload = {"node": node.model_dump(mode="json"), "files": _file_digests(node),
               "inputs": dict(inputs), "temperature": 0, "settings": dict(settings or {})}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class _Entry(NamedTuple):
    expires: float | None  # time.monotonic() deadline; None never expires
    entity: str | None     # whose writes invalidate a read; None when unknown
    read: bool
    output: dict[str, Any]


class ExactCache:
    """Node results by cache_key. Reads expire after their ttl_s and are dropped when a write
    may have changed them; pure results stay until the cache is discarded."""

    def __init__(self) -> None:
        self.entries: dict[str, _Entry] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        """A copy of the cached output, or None when absent or expired."""
        entry = self.entries.get(key)
        if entry is None:
            return None
        if entry.expires is not None and time.monotonic() >= entry.expires:
            del self.entries[key]
            return None
        return json.loads(json.dumps(entry.output))

    def put(self, key: str, output: dict[str, Any], ttl_s: int | None, entity: str | None,
            *, read: bool | None = None) -> None:
        """Store a copy of a node's output. A read must have ttl_s; entity names the record
        whose writes invalidate it. read defaults to whether ttl_s or entity is given."""
        read = ttl_s is not None or entity is not None if read is None else read
        if read and ttl_s is None:
            raise ValueError("a read result can be cached only within its ttl_s")
        expires = time.monotonic() + ttl_s if ttl_s is not None else None
        self.entries[key] = _Entry(expires, entity, read, json.loads(json.dumps(output)))

    def invalidate(self, entity: str | None) -> None:
        """Drop the cached reads a write may have changed: those of the written entity and
        those whose entity is unknown, or every read when the write's entity is unknown."""
        for key, entry in list(self.entries.items()):
            if entry.read and (entity is None or entry.entity in (None, entity)):
                del self.entries[key]


def cache_blocker(node: Node) -> str | None:
    """Why a node's result must not be reused, or None when memoising it is legal."""
    if node.effect is Effect.WRITE:
        return "write effect"
    if node.effect is Effect.READ and node.ttl_s is None:
        return "read without ttl_s, so its freshness is unknown"
    if node.kind is not NodeKind.LLM and not node.deterministic:
        return "not deterministic"
    return None


class Memoise:
    name = "memoise"

    def plan(self, wf: Workflow, profile: Profile | None = None) -> list[Decision]:
        """One decision per node: applied when its result may be reused, else skipped."""
        decisions = []
        for node in wf.nodes:
            blocker = cache_blocker(node)
            if blocker:
                status, reason = "skipped", f"{node.id} not cached: {blocker}"
            elif node.effect is Effect.READ:
                status, reason = "applied", (f"{node.id} cached for {node.ttl_s} s: exact key; "
                                             "a later write invalidates cached reads")
            else:
                status, reason = "applied", f"{node.id} cached: pure result under an exact key"
            decisions.append(Decision(pass_name=self.name, nodes=[node.id], status=status,
                                      reason=reason))
        return decisions

    def apply(self, wf: Workflow, decisions: list[Decision]) -> Workflow:
        """The workflow unchanged, after checking that every node chosen for caching may be."""
        by_id = {node.id: node for node in wf.nodes}
        for decision in decisions:
            if decision.status != "applied":
                continue
            for node_id in decision.nodes:
                if node_id not in by_id:
                    raise ValueError(f"cannot cache {node_id}: no such node")
                blocker = cache_blocker(by_id[node_id])
                if blocker:
                    raise ValueError(f"cannot cache {node_id}: {blocker}")
        return wf.model_copy(deep=True)
