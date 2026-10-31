"""Tool registry and the HTTP client for the mock bank APIs.

A tool node's inputs become the JSON request body, each named by its last path segment
(applicant.pan -> pan); the JSON response is the node's output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

# Tool name used in workflows -> endpoint on the bank API.
TOOLS = {
    "kyc_api": "/kyc",
    "bureau_api": "/bureau",
    "statement_api": "/statement",
    "offers_api": "/offers",
    "sms_gateway": "/sms",
    "transactions_api": "/transactions",
    "ticket_api": "/tickets",
}


def tool_arguments(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The request body for a tool call: each input named by its last path segment."""
    arguments: dict[str, Any] = {}
    for reference, value in inputs.items():
        name = reference.rsplit(".", 1)[-1]
        if name in arguments:
            raise ValueError(f"two inputs end in {name!r}, so the tool cannot tell them apart")
        arguments[name] = value
    return arguments


class ToolClient:
    """Calls bank API tools through an httpx client, at a real URL or on an in-process app."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def call(self, tool: str, inputs: Mapping[str, Any]) -> Any:
        """POST a tool's arguments to its endpoint and return the JSON response."""
        if tool not in TOOLS:
            raise LookupError(f"unknown tool {tool!r}")
        response = await self._client.post(TOOLS[tool], json=tool_arguments(inputs))
        response.raise_for_status()
        return response.json()
