"""Workflow IR and run records: the contracts between team members.

Change these only by agreement. The Pass protocol lives in passes/base.py, LLMAdapter in
backends/llm.py, run() in executor.py, and Check and validate() in validator.py.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Effect(str, Enum):
    PURE = "pure"    # no outside contact
    READ = "read"    # reads outside data
    WRITE = "write"  # changes the outside world


class NodeKind(str, Enum):
    LLM = "llm"
    RETRIEVE = "retrieve"
    TOOL = "tool"
    TRANSFORM = "transform"
    GATE = "gate"


class Node(BaseModel):
    id: str
    kind: NodeKind
    inputs: list[str] = Field(default_factory=list)   # e.g. "applicant.pan"
    outputs: list[str] = Field(default_factory=list)
    effect: Effect = Effect.WRITE                     # unknown means write
    deterministic: bool = False
    required: bool = False  # mandatory check even when no consumer reads its output
    ttl_s: int | None = None
    model: Literal["small", "large"] | None = None
    fallback_model: Literal["large"] | None = None
    prompt: str | None = None                         # path under prompts/
    tool: str | None = None
    index: str | None = None                          # retrieve nodes
    query: str | None = None
    tests: list[dict[str, Any]] = Field(default_factory=list)
    compact_inputs: bool = False
    max_output_tokens: int = Field(default=1024, ge=128, le=8192)


class Workflow(BaseModel):
    name: str
    inputs: list[str]
    nodes: list[Node]
    outputs: list[str] = Field(default_factory=list)  # required results; empty means unknown


class Usage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    model: str | None = None
    cost: float = 0.0          # cost units in mock mode, USD in live mode
    retries: int = 0           # rate-limit retries a live call needed
    calls: int = 0
    quota_wait_ms: float = 0.0
    retry_wait_ms: float = 0.0
    provider_ms: float = 0.0
    usage_complete: bool = True


class NodeRun(BaseModel):
    run_id: str
    node_id: str
    start_s: float             # seconds since run start
    end_s: float
    usage: Usage = Field(default_factory=Usage)
    cache_hit: bool = False
    input_hash: str | None = None
    retries: int = 0
    cpu_pct: float | None = None
    mem_mb: float | None = None
    error: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    run_id: str
    workflow: str
    mode: Literal["baseline", "compiled"]
    plan_id: str | None = None
    case_id: str
    live: bool = False
    cold_cache: bool = True
    latency_scale: float | None = None
    wall_ms: float
    nodes: list[NodeRun]


class NodeStats(BaseModel):
    node_id: str
    p50_ms: float
    p95_ms: float
    tokens_in: float
    tokens_out: float
    cost: float
    repeat_input_rate: float   # share of runs whose input hash was seen before


class Profile(BaseModel):
    workflow: str
    runs: int
    nodes: dict[str, NodeStats]
    critical_path: list[str]


class Decision(BaseModel):
    pass_name: str
    nodes: list[str]
    reason: str
    status: Literal["applied", "skipped", "rejected"]
    predicted_ms: float | None = None
    predicted_cost: float | None = None
    measured_ms: float | None = None
    measured_cost: float | None = None


class ExecPlan(BaseModel):
    plan_id: str
    workflow: Workflow                                   # the rewritten workflow
    order_edges: list[tuple[str, str]] = Field(default_factory=list)  # write-order chain
    cached_nodes: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
