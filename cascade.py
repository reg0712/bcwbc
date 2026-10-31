"""Cascade with escalation, the runtime half of model routing.

A routed LLM node (model small, fallback_model large) tries the small model first. When its
output fails the node's check, the executor retries on the large model; when that fails too,
the run stops and asks for human review. The cascade is legal only when the output has a
checkable test, because nothing else could tell the executor that the small answer is wrong.
"""

from __future__ import annotations

from synapse.ir import Node


def checkable(node: Node) -> bool:
    """Whether every output of the node is checked as it is produced: a schema test, which the
    executor enforces on every model call."""
    return any("schema" in test for test in node.tests)
