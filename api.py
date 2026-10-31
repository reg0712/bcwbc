"""SYNAPSE API and dashboard server. The dashboard (ui/) builds against these routes:

    GET  /workflows                        workflow files under workflows/
    POST /compile  {workflow}              the ExecPlan, its decisions per pass, a graph view
    POST /runs     {workflow, mode, case_id, live}   starts a run and returns its run_id
    GET  /runs/{run_id}                    the RunRecord (HTTP 202 while it is running)
    WS   /ws/runs/{run_id}                 node_start, node_end and run_end events
    POST /validate {workflow, plan_id}     checks and pass verdicts
    GET  /bench/latest                     the latest benchmark JSON (?mode=live for live)

Runs started here are kept in memory; GET /runs falls back to the SQLite store for runs the
CLI saved. Profiles come only from stored baseline runs, so dashboard runs at demo speed
never skew the optimiser's predictions.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bench.runner import RESULTS_DIR, STEPS, Target, cost_unit, load_target, results_path
from synapse import ROOT
from synapse.backends import open_backends
from synapse.backends.quota import LIMITS, QuotaGuard
from synapse.compiler import compile_workflow
from synapse.config import load_config
from synapse.executor import execute, final_outputs, resolve
from synapse.graph import build_graph, dependencies, levels
from synapse.ir import Decision, ExecPlan, NodeKind, Profile, RunRecord, Usage, Workflow
from synapse.optimiser import optimise
from synapse.profiler import build_profile
from synapse.store import Store

PROFILE_RUNS = 3        # baseline runs measured when a workflow has no stored profile
PROFILE_SCALE = 0.01    # the same mock scale `make demo` profiles at
VALIDATE_SCALE = 0.001  # validation checks outputs, so delays only cost time
MAX_RUNS = 500          # finished runs kept in memory
PASS_ORDER = [step.name for step in STEPS[1:]]
WORKFLOW_SUFFIXES = (".yaml", ".yml", ".py")


class CompileRequest(BaseModel):
    workflow: str


class RunRequest(BaseModel):
    workflow: str
    mode: Literal["baseline", "compiled"] = "baseline"
    case_id: str | None = None          # default: the first test case
    live: bool = False
    provider: Literal["mock", "gemini", "groq"] | None = None
    model: str | None = None
    plan_id: str | None = None          # default: the workflow's latest compiled plan
    cache_mode: Literal["cold", "warm"] = "cold"
    scale: float | None = Field(default=None, ge=0)  # mock delays; default config/mock.yaml


class ValidateRequest(BaseModel):
    workflow: str
    plan_id: str | None = None          # also check this plan's outputs case by case
    live: bool = False
    unsafe_route_risk: bool = False
    scale: float | None = Field(default=None, ge=0)


class _UsageTap:
    """Wraps a run's LLM adapter to keep each node's latest usage, so node_end events carry
    tokens and cost while the run is still going."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.usage: dict[str, Usage] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def invoke(self, node: Any, prompt: str, schema: Any) -> tuple[dict[str, Any], Usage]:
        result, usage = await self.inner.invoke(node, prompt, schema)
        self.usage[node.id] = usage  # the executor finalises this object's cost in place
        return result, usage


class _Delayed:
    """A shared retriever with one run's simulated vector-store round trip in front, so the
    embedding model loads once while every run keeps its own latency scale."""

    def __init__(self, inner: Any, delay_s: float) -> None:
        self.inner = inner
        self.delay_s = delay_s

    async def warm(self, indexes: Any) -> None:
        await self.inner.warm(indexes)

    async def search(self, index: str, query: str, k: int = 3) -> list[dict[str, Any]]:
        await asyncio.sleep(self.delay_s)
        return await self.inner.search(index, query, k)


@dataclass
class RunState:
    """One run started through the API: its buffered events, then its record."""

    run_id: str
    file: str
    name: str
    mode: str
    case_id: str
    live: bool
    scale: float
    plan_id: str | None
    target: Target
    provider: str | None = None
    model: str | None = None
    cache_mode: str = "cold"
    events: list[dict[str, Any]] = field(default_factory=list)
    record: RunRecord | None = None
    error: str | None = None
    done: bool = False
    task: asyncio.Task[None] | None = None
    _wake: asyncio.Event = field(default_factory=asyncio.Event)

    def publish(self, event: dict[str, Any]) -> None:
        """Buffer an event and wake every follower."""
        self.events.append(event)
        self._wake.set()
        self._wake = asyncio.Event()

    def finish(self, record: RunRecord | None, error: str | None = None) -> None:
        """Record the outcome and send run_end."""
        self.record, self.error = record, error
        failure = error
        if record is not None and failure is None:
            failure = next((node.error for node in record.nodes if node.error), None)
        self.publish({"type": "run_end", "run_id": self.run_id,
                      "wall_ms": record.wall_ms if record else 0.0, "error": failure})
        self.done = True

    async def follow(self) -> AsyncIterator[dict[str, Any]]:
        """Every event so far, then each new one until run_end."""
        sent = 0
        while True:
            while sent < len(self.events):
                yield self.events[sent]
                sent += 1
            if self.done:
                return
            await self._wake.wait()

    def meta(self) -> dict[str, Any]:
        """What a client needs to label the run."""
        return {"run_id": self.run_id, "workflow": self.name, "file": self.file,
                "mode": self.mode, "case_id": self.case_id, "live": self.live,
                "scale": self.scale, "plan_id": self.plan_id, "provider": self.provider,
                "model": self.model, "cache_mode": self.cache_mode,
                "cost_unit": cost_unit(self.provider) if self.provider else None}


def contract_event(run_id: str, event: Mapping[str, Any],
                   usage: Usage | None = None) -> dict[str, Any] | None:
    """An executor event in the API's event format; None for events the API does not send.
    Token and cost fields come from the event when the executor sends them, else from the
    usage the LLM tap saw."""
    kind = event.get("type")
    t_ms = float(event["t_ms"]) if "t_ms" in event else float(event.get("at_s", 0.0)) * 1000
    if kind == "node_start":
        return {"type": "node_start", "run_id": run_id, "node_id": event["node_id"], "t_ms": t_ms}
    if kind in ("node_wait", "node_resume"):
        return {**event, "run_id": run_id, "t_ms": t_ms}
    if kind not in ("node_end", "node_finish"):
        return None
    sent = event.get("usage") if isinstance(event.get("usage"), Mapping) else event
    def pick(name: str, fallback: Any) -> Any:
        return sent[name] if name in sent else fallback
    return {"type": "node_end", "run_id": run_id, "node_id": event["node_id"], "t_ms": t_ms,
            "cache_hit": bool(event.get("cache_hit", False)),
            "tokens_in": int(pick("tokens_in", usage.tokens_in if usage else 0)),
            "tokens_out": int(pick("tokens_out", usage.tokens_out if usage else 0)),
            "cost": float(pick("cost", usage.cost if usage else 0.0)),
            "error": event.get("error")}


def replay_events(record: RunRecord, run_id: str | None = None) -> list[dict[str, Any]]:
    """The events a finished record would have sent, in time order (for stored runs)."""
    run_id = run_id or record.run_id
    events: list[tuple[float, int, dict[str, Any]]] = []
    for node in record.nodes:
        events.append((node.start_s, 0, contract_event(run_id, {
            "type": "node_start", "node_id": node.node_id, "at_s": node.start_s})))
        events.append((node.end_s, 1, contract_event(run_id, {
            "type": "node_end", "node_id": node.node_id, "at_s": node.end_s,
            "cache_hit": node.cache_hit, "error": node.error}, node.usage)))
    events.sort(key=lambda item: (item[0], item[1]))
    failure = next((node.error for node in record.nodes if node.error), None)
    return [event for _, _, event in events] + [
        {"type": "run_end", "run_id": run_id, "wall_ms": record.wall_ms, "error": failure}]


def usage_totals(record: RunRecord) -> dict[str, int | float]:
    """Measured model usage for one run, including retries and any fallback calls."""
    tokens_in = sum(node.usage.tokens_in for node in record.nodes)
    tokens_out = sum(node.usage.tokens_out for node in record.nodes)
    return {"tokens_in": tokens_in, "tokens_out": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "cost": sum(node.usage.cost for node in record.nodes),
            "calls": sum(node.usage.calls for node in record.nodes),
            "quota_wait_ms": sum(node.usage.quota_wait_ms for node in record.nodes),
            "retry_wait_ms": sum(node.usage.retry_wait_ms for node in record.nodes),
            "provider_ms": sum(node.usage.provider_ms for node in record.nodes),
            "usage_complete": all(node.usage.usage_complete for node in record.nodes)}


def graph_view(source: Workflow, plan: ExecPlan) -> dict[str, Any]:
    """Both graphs for the dashboard: the original in file order (the order the baseline
    runs it), the compiled one in the levels it runs in, with removed, merged, routed and
    cached nodes marked."""
    applied = [d for d in plan.decisions if d.status == "applied"]
    parallel = any(d.pass_name == "parallelise" for d in applied)
    removed: dict[str, dict[str, Any]] = {}
    for decision in applied:
        if decision.pass_name == "dead_nodes":
            removed.update({id: {"pass": "dead_nodes", "into": None} for id in decision.nodes})
        elif decision.pass_name == "cse" and len(decision.nodes) == 2:
            removed[decision.nodes[1]] = {"pass": "cse", "into": decision.nodes[0]}
    original = build_graph(source)
    compiled = build_graph(plan.workflow)
    schedule = compiled.copy()  # order edges carry no data, so dependencies() reads compiled
    schedule.add_edges_from(plan.order_edges)
    before = {node.id: node for node in source.nodes}
    groups = (levels(schedule) if parallel
              else [[node.id] for node in plan.workflow.nodes])
    level_of = {id: number for number, group in enumerate(groups) for id in group}

    def describe(node: Any, graph: Any) -> dict[str, Any]:
        return {"id": node.id, "kind": node.kind.value, "effect": node.effect.value,
                "model": node.model, "fallback_model": node.fallback_model,
                "deps": dependencies(graph, node.id)}

    source_nodes = [{**describe(node, original), "removed": removed.get(node.id)}
                    for node in source.nodes]
    compiled_nodes = []
    for node in plan.workflow.nodes:
        old = before.get(node.id)
        routed = old is not None and old.model != node.model
        compiled_nodes.append({
            **describe(node, compiled), "level": level_of.get(node.id, 0),
            "cached": node.id in plan.cached_nodes,
            "routed": {"from": old.model, "to": node.model,
                       "fallback": node.fallback_model} if routed else None,
            "merged": sorted(id for id, why in removed.items() if why["into"] == node.id),
            "barrier": node.effect.value == "write"})
    return {"parallel": parallel, "levels": groups, "removed": removed,
            "source": {"nodes": source_nodes},
            "compiled": {"nodes": compiled_nodes,
                         "order_edges": [list(edge) for edge in plan.order_edges]}}


def _bench_cost(row: Mapping[str, Any]) -> float | None:
    """A configuration's mean cost (older result files name it mean_cost_mock_units)."""
    value = row.get("mean_cost", row.get("mean_cost_mock_units"))
    return float(value) if value is not None else None


def measured_gains(result: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Each pass's measured gain from a benchmark: the cold median before the step minus
    after it; caching is measured warm (repeat customer), since a cold cache cannot help."""
    if not result:
        return {}
    rows = result.get("configurations", [])
    cold = {row["name"]: row for row in rows if row.get("cache") == "cold"}
    warm = {row["name"]: row for row in rows if row.get("cache") == "warm"}
    order = [step["name"] for step in result.get("steps", [])] or list(cold)
    baseline = cold.get(order[0]) if order else None
    gains: dict[str, dict[str, Any]] = {}
    for previous, name in zip(order, order[1:]):
        if previous not in cold or name not in cold:
            continue
        basis = "warm" if name == "memoise" and name in warm else "cold"
        after = warm[name] if basis == "warm" else cold[name]
        before = cold[previous]
        saved_ms = before["p50_ms"] - after["p50_ms"]
        cost_before, cost_after = _bench_cost(before), _bench_cost(after)
        gains[name] = {
            "measured_ms": saved_ms,
            "measured_share": saved_ms / baseline["p50_ms"] if baseline else None,
            "measured_cost": (cost_before - cost_after
                              if cost_before is not None and cost_after is not None else None),
            "measured_basis": basis,
            "measured_cold_ms": before["p50_ms"] - cold[name]["p50_ms"]}
    return gains


def pass_rows(plan: ExecPlan, profile: Profile | None, gains: Mapping[str, Mapping[str, Any]],
              verdicts: Mapping[str, Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """One decision-log row per pass: status, reasons, nodes, predicted and measured gains,
    and the latest validation verdict."""
    names = list(PASS_ORDER)
    names += [d.pass_name for d in plan.decisions if d.pass_name not in names]
    names = list(dict.fromkeys(names))
    serial_ms = sum(stat.p50_ms for stat in profile.nodes.values()) if profile else None
    rows = []
    for name in names:
        decisions = [d for d in plan.decisions if d.pass_name == name]
        statuses = {d.status for d in decisions}
        status = ("rejected" if "rejected" in statuses else "applied" if "applied" in statuses
                  else "skipped" if decisions else "no change")
        shown = [d for d in decisions if d.status == status] or decisions
        predicted = [d.predicted_ms for d in decisions if d.predicted_ms is not None]
        predicted_cost = [d.predicted_cost for d in decisions if d.predicted_cost is not None]
        predicted_ms = sum(predicted) if predicted else None
        rows.append({
            "pass_name": name, "status": status,
            "label": next((s.label for s in STEPS if s.name == name), name),
            "reason": "; ".join(dict.fromkeys(d.reason for d in shown)),
            "nodes": list(dict.fromkeys(id for d in shown for id in d.nodes)),
            "decisions": [d.model_dump(mode="json") for d in decisions],
            "predicted_ms": predicted_ms,
            "predicted_share": (predicted_ms / serial_ms
                                if predicted_ms is not None and serial_ms else None),
            "predicted_cost": sum(predicted_cost) if predicted_cost else None,
            **{key: None for key in ("measured_ms", "measured_share", "measured_cost",
                                     "measured_basis", "measured_cold_ms")},
            **gains.get(name, {}),
            "verdict": (verdicts or {}).get(name)})
    return rows


def pass_verdicts(plan: ExecPlan) -> dict[str, dict[str, Any]]:
    """Validation verdict per pass in a validated plan: rejected when a check failed,
    accepted when the validator kept it."""
    verdicts: dict[str, dict[str, Any]] = {}
    for decision in plan.decisions:
        if decision.status == "rejected":
            verdicts[decision.pass_name] = {"verdict": "rejected", "reason": decision.reason,
                                            "nodes": decision.nodes}
        elif decision.status == "applied":
            verdicts[decision.pass_name] = {"verdict": "accepted", "reason": decision.reason,
                                            "nodes": decision.nodes}
        elif decision.pass_name not in verdicts:
            verdicts[decision.pass_name] = {"verdict": "no change", "reason": decision.reason,
                                            "nodes": []}
    return verdicts


def case_results(checks: Sequence[Any], cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Checks grouped by test case, in suite order."""
    by_case: dict[str, list[Any]] = {}
    for check in checks:
        by_case.setdefault(check.case_id, []).append(check)
    rows = []
    for case in cases:
        own = by_case.get(str(case["case_id"]), [])
        failed = [c for c in own if not c.passed]
        rows.append({"case_id": str(case["case_id"]), "checks": len(own),
                     "passed": bool(own) and not failed,
                     "failed": [{"node_id": c.node_id, "kind": c.kind, "detail": c.detail}
                                for c in failed]})
    return rows


class Lab:
    """The server's state: workflow targets, compiled plans, runs and validations."""

    def __init__(self, workflows_dir: Path, db_path: Path | None, results_dir: Path,
                 backend_options: Mapping[str, Any] | None) -> None:
        self.workflows_dir = workflows_dir
        self.db_path = db_path
        self.quota_path = db_path.with_name("quota.db") if db_path else ROOT / "data" / "quota.db"
        self.results_dir = results_dir
        self.backend_options = dict(backend_options or {})
        self.plans: dict[str, tuple[str, ExecPlan]] = {}
        self.current: dict[str, str] = {}
        self.runs: dict[str, RunState] = {}
        self.validations: dict[str, dict[str, Any]] = {}
        self._targets: dict[str, tuple[float, Target]] = {}
        self._retriever: Any = None
        self._retriever_lock = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._run_caches: dict[tuple, Any] = {}
        self._cache_locks: dict[tuple, asyncio.Lock] = {}

    def store(self) -> Store:
        return Store(self.db_path) if self.db_path else Store()

    def files(self) -> list[Path]:
        """Workflow files: YAML workflows and LangGraph modules."""
        return sorted(path for path in self.workflows_dir.iterdir()
                      if path.suffix in WORKFLOW_SUFFIXES and not path.name.startswith("_"))

    def load(self, path: Path) -> Target:
        """A workflow file as a Target, reloaded when the file changes."""
        mtime = path.stat().st_mtime
        cached = self._targets.get(path.name)
        if cached is None or cached[0] != mtime:
            self._targets[path.name] = (mtime, load_target(path))
        return self._targets[path.name][1]

    def target(self, name: str) -> tuple[str, Target]:
        """Find a workflow by file name, file stem or workflow name; 404 if none."""
        files = self.files()
        for path in files:
            if name in (path.name, path.stem):
                return path.name, self._load_or_422(path)
        for path in files:
            try:
                if self.load(path).workflow.name == name:
                    return path.name, self.load(path)
            except Exception:  # a broken file cannot be the one asked for by name
                continue
        raise HTTPException(404, f"no workflow {name!r}; see GET /workflows")

    def _load_or_422(self, path: Path) -> Target:
        try:
            return self.load(path)
        except Exception as exc:
            raise HTTPException(422, f"{path.name}: {exc}") from exc

    def lock(self, name: str) -> asyncio.Lock:
        return self._locks.setdefault(name, asyncio.Lock())

    def plan(self, plan_id: str, workflow: str) -> ExecPlan:
        """A plan this server compiled or validated for the workflow; 404 otherwise."""
        owner, plan = self.plans.get(plan_id, (None, None))
        if plan is None or owner != workflow:
            raise HTTPException(404, f"no plan {plan_id!r} for {workflow}; POST /compile first")
        return plan

    def remember(self, workflow: str, plan: ExecPlan, current: bool = True) -> None:
        self.plans[plan.plan_id] = (workflow, plan)
        if current:
            self.current[workflow] = plan.plan_id

    async def retriever(self, scale: float) -> Any:
        """The configured retriever; the embedding model loads once per server."""
        if "retriever" in self.backend_options:
            return self.backend_options["retriever"]
        async with self._retriever_lock:
            if self._retriever is None:
                from synapse.backends.retrieval import EmbeddingRetriever
                self._retriever = await asyncio.to_thread(EmbeddingRetriever)
        delay_ms = load_config("mock.yaml")["retrieve_latency_ms"] * scale
        return _Delayed(self._retriever, delay_ms / 1000)

    @asynccontextmanager
    async def backends(self, target: Target, live: bool, scale: float,
                       provider: str | None = None, model: str | None = None
                       ) -> AsyncIterator[Any]:
        """Backends for one run or validation, with the workflow's imported functions."""
        options = {**self.backend_options, "retriever": await self.retriever(scale)}
        if live:
            options["quota_path"] = self.quota_path
        else:
            options.pop("llm", None)  # injected live adapter must not change mock profiling
        if target.functions:
            options["functions"] = target.functions
        async with open_backends(live, scale, provider=provider, model=model,
                                 **options) as backends:
            await backends.retriever.warm({n.index for n in target.workflow.nodes if n.index})
            yield backends

    async def profile(self, target: Target) -> tuple[Profile | None, dict[str, Any] | None]:
        """The profile from stored mock baseline runs; measures PROFILE_RUNS runs of the
        first case (and stores them) when there are none."""
        wf = target.workflow
        with self.store() as store:
            records = [store.load_run(id) for id in store.list_runs(wf.name)]
        records = [r for r in records if r.mode == "baseline" and not r.live
                   and not any(node.error for node in r.nodes)]
        if records:
            latest_scale = records[-1].latency_scale
            records = [r for r in records if r.latency_scale == latest_scale]
        auto = not records
        if auto and not target.cases:
            return None, None
        if auto:
            async with self.backends(target, False, PROFILE_SCALE) as backends:
                records = [await execute(wf, target.cases[0], "baseline", backends)
                           for _ in range(PROFILE_RUNS)]
            with self.store() as store:
                for record in records:
                    store.save_run(record)
        profile = build_profile(wf, records)
        serial = sum(stat.p50_ms for stat in profile.nodes.values())
        return profile, {"runs": profile.runs, "auto": auto,
                         "scale": PROFILE_SCALE if auto else None, "serial_ms": serial,
                         "critical_path": profile.critical_path,
                         "cases": sorted({r.case_id for r in records})}

    async def compile(self, name: str, target: Target) -> dict[str, Any]:
        """Optimise the workflow from its profile, record the decisions, keep the plan."""
        wf = target.workflow
        async with self.lock(name):
            profile, profile_info = await self.profile(target)
            with self.store() as store:
                plan = optimise(wf, profile, validated_routing=store.route_validated(wf))
                store.save_decisions(plan)
        self.remember(name, plan)
        bench = self.bench(wf.name)
        validation = self.validations.get(name)
        return {"workflow": name, "name": wf.name, "plan_id": plan.plan_id,
                "source": wf.model_dump(mode="json"), "plan": plan.model_dump(mode="json"),
                "profile": profile_info, "graph": graph_view(wf, plan),
                "passes": pass_rows(plan, profile, measured_gains(bench),
                                    validation["verdicts"] if validation else None),
                "bench": {key: bench.get(key) for key in (
                    "created_at", "latency_scale", "repeats", "cases", "mode")} if bench else None,
                "validation": {key: validation[key] for key in ("plan_id", "accepted_plan_id",
                               "unsafe_route_risk")} if validation else None}

    async def current_plan(self, name: str, target: Target) -> ExecPlan:
        """The plan compiled last for this workflow, compiling one if there is none."""
        if name not in self.current:
            await self.compile(name, target)
        return self.plans[self.current[name]][1]

    def live_plan(self, target: Target, plan: ExecPlan, *, explicit: bool,
                  provider: str | None = None, model: str | None = None) -> ExecPlan:
        """Use routed nodes live only after that provider and model pair passed validation."""
        if not any(node.fallback_model for node in plan.workflow.nodes):
            return plan
        models = load_config("models.yaml")
        provider = provider or models["live_provider"]
        tiers = models["providers"][provider]
        names = ({"small": model, "large": model} if model else
                 {tier: str(tiers[tier]) for tier in ("small", "large")})
        with self.store() as store:
            if store.route_validated(target.workflow, provider=provider, model_names=names, plan=plan):
                return plan
        if explicit:
            raise HTTPException(422, "routed plan has not passed validation on the live models")
        selected = tuple(name for name in ("dead_nodes", "cse", "compact_inputs", "parallelise", "memoise")
                         if any(decision.pass_name == name for decision in plan.decisions))
        return compile_workflow(target.workflow, passes=selected)

    def bench(self, workflow: str, live: bool = False) -> dict[str, Any] | None:
        """The latest benchmark result for one workflow, if any."""
        path = results_path(live, self.results_dir)
        if not path.exists():
            return None
        results = json.loads(path.read_text())
        return next((r for r in results if r.get("workflow") == workflow), None)

    def start(self, name: str, target: Target, case: dict[str, Any], mode: str,
              plan: ExecPlan | None, live: bool, scale: float,
              provider: str | None = None, model: str | None = None,
              cache_mode: str = "cold") -> RunState:
        """Start a run in the background; its events buffer until someone follows them."""
        run = RunState(run_id=f"r{uuid.uuid4().hex[:12]}", file=name,
                       name=target.workflow.name, mode=mode, case_id=str(case["case_id"]),
                       live=live, scale=scale, plan_id=plan.plan_id if plan else None,
                       target=target, provider=provider if live else "mock", model=model,
                       cache_mode=cache_mode if mode == "compiled" else "cold")
        finished = [id for id, state in self.runs.items() if state.done]
        for id in finished[:max(0, len(self.runs) - MAX_RUNS + 1)]:
            del self.runs[id]
        self.runs[run.run_id] = run
        run.task = asyncio.create_task(self._execute(run, case, plan))
        return run

    async def _execute(self, run: RunState, case: dict[str, Any], plan: ExecPlan | None) -> None:
        """Serialize repeated warm runs so concurrent clicks cannot duplicate model work."""
        from synapse.passes.memoise import ExactCache
        if run.cache_mode == "warm" and plan is not None:
            key = (plan.plan_id, run.provider, run.model)
            if key not in self._run_caches:
                if len(self._run_caches) >= 32:
                    await self._execute_record(run, case, plan, ExactCache())
                    return
                self._run_caches[key] = ExactCache()
                self._cache_locks[key] = asyncio.Lock()
            async with self._cache_locks[key]:
                await self._execute_record(run, case, plan, self._run_caches[key])
        else:
            await self._execute_record(run, case, plan)

    async def _execute_record(self, run: RunState, case: dict[str, Any],
                              plan: ExecPlan | None, cache=None) -> None:
        """Execute with fresh services and an optional exact result cache."""
        wf = run.target.workflow
        try:
            async with self.backends(run.target, run.live, run.scale,
                                     run.provider if run.live else None, run.model) as backends:
                run.provider = getattr(backends, "provider", "live" if run.live else "mock")
                tap = _UsageTap(backends.llm)
                backends.llm = tap

                async def on_event(event: dict[str, Any]) -> None:
                    usage = tap.usage.pop(event.get("node_id", ""), None) \
                        if event.get("type") in ("node_end", "node_finish") else None
                    translated = contract_event(run.run_id, event, usage)
                    if translated is not None:
                        run.publish(translated)

                record = await execute(wf, case, run.mode, backends, plan=plan,  # type: ignore[arg-type]
                                       live=run.live, on_event=on_event, cache=cache)
        except Exception as exc:  # the run never finished; its followers still get run_end
            run.finish(None, f"{type(exc).__name__}: {exc}")
            return
        record = record.model_copy(update={
            "run_id": run.run_id,
            "nodes": [node.model_copy(update={"run_id": run.run_id}) for node in record.nodes]})
        try:
            with self.store() as store:
                store.save_run(record)
        except Exception as exc:
            run.finish(record, f"Could not persist run: {type(exc).__name__}")
            return
        run.finish(record)

    def run_body(self, run: RunState) -> dict[str, Any]:
        """GET /runs/{id}: the RunRecord, plus status and the decision it reached."""
        body: dict[str, Any] = {**run.meta(), "status": "running"}
        if run.record is None:
            return {**body, "status": "failed" if run.done else "running", "error": run.error,
                    "events": len(run.events)}
        body.update(run.record.model_dump(mode="json"))
        body["usage_totals"] = usage_totals(run.record)
        failed_node = run.error or next((node.error for node in run.record.nodes if node.error), None)
        body["status"] = "failed" if failed_node else "done"
        body["error"] = failed_node
        target = run.target
        if target.key_field:
            try:
                body["key_value"] = resolve(target.key_field, final_outputs(run.record))
            except KeyError:
                body["key_value"] = None
            body["key_field"] = target.key_field
            body["expected"] = target.expected(run.case_id) if target.expected else None
        return body

    async def validate(self, name: str, target: Target, request: ValidateRequest) -> dict[str, Any]:
        """Validate pass by pass on every case; with plan_id, also check that plan."""
        from synapse.validator import validate, validate_workflow

        wf = target.workflow
        requested = self.plan(request.plan_id, name) if request.plan_id else None
        scale = VALIDATE_SCALE if request.scale is None else request.scale
        started = time.perf_counter()
        async with self.backends(target, request.live, scale) as backends:
            try:
                accepted, checks, floor = await validate_workflow(
                    wf, target.cases, backends, unsafe_route_risk=request.unsafe_route_risk,
                    live=request.live)
            except (LookupError, StopIteration, ValueError) as exc:
                raise HTTPException(422, f"cannot validate {name}: {exc}") from exc
            accepted_checks = checks
            proof_provider, proof_models = backends.provider, backends.model_names.copy()
            checked = accepted
            if requested is not None and requested.plan_id != accepted.plan_id:
                golden = [await execute(wf, case, "baseline", backends, live=request.live)
                          for case in target.cases]
                candidate = [await execute(wf, case, "compiled", backends, plan=requested,
                                           live=request.live) for case in target.cases]
                checks, checked = validate(
                    golden, candidate, floor, wf=wf,
                    key_fields=[target.key_field] if target.key_field else []), requested
        with self.store() as store:
            store.save_decisions(accepted)
            store.save_validation(wf, accepted,
                                  [str(case["case_id"]) for case in target.cases],
                                  bool(accepted_checks) and
                                  all(check.passed for check in accepted_checks),
                                  provider=proof_provider, model_names=proof_models)
        self.remember(name, accepted, current=False)
        verdicts = pass_verdicts(accepted)
        result = {"workflow": name, "name": wf.name, "plan_id": checked.plan_id,
                  "accepted_plan_id": accepted.plan_id, "noise_floor": floor, "scale": scale,
                  "unsafe_route_risk": request.unsafe_route_risk,
                  "duration_s": round(time.perf_counter() - started, 2),
                  "checks": [check.model_dump(mode="json") for check in checks],
                  "passed": sum(check.passed for check in checks),
                  "cases": case_results(checks, target.cases),
                  "verdicts": verdicts,
                  "plan": accepted.model_dump(mode="json")}
        self.validations[name] = result
        return result


def create_app(*, workflows_dir: str | Path | None = None, db_path: str | Path | None = None,
               results_dir: str | Path | None = None,
               backend_options: Mapping[str, Any] | None = None) -> FastAPI:
    """The API with its own state. Tests pass a temporary database and results directory,
    and backend options such as a KeywordRetriever."""
    lab = Lab(Path(workflows_dir) if workflows_dir else ROOT / "workflows",
              Path(db_path) if db_path else None,
              Path(results_dir) if results_dir else ROOT / "bench" / "results",
              backend_options)
    app = FastAPI(title="SYNAPSE", description=__doc__)
    app.state.lab = lab
    ui = ROOT / "ui"
    app.mount("/ui", StaticFiles(directory=ui), name="ui")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(ui / "index.html")

    @app.get("/workflows")
    def workflows() -> list[dict[str, Any]]:
        """Every workflow file, with its cases; files that do not load carry an error."""
        listed = []
        for path in lab.files():
            entry: dict[str, Any] = {"file": path.name,
                                     "kind": "langgraph" if path.suffix == ".py" else "yaml"}
            try:
                target = lab.load(path)
            except Exception as exc:
                listed.append({**entry, "name": path.stem, "error": f"{type(exc).__name__}: {exc}"})
                continue
            wf = target.workflow
            listed.append({**entry, "name": wf.name, "error": None, "nodes": len(wf.nodes),
                           "inputs": wf.inputs, "outputs": wf.outputs,
                           "cases": [str(case["case_id"]) for case in target.cases],
                           "key_field": target.key_field})
        return listed

    @app.post("/compile")
    async def compile_route(request: CompileRequest) -> dict[str, Any]:
        """Optimise a workflow from its measured profile; returns the plan and its decisions."""
        name, target = lab.target(request.workflow)
        return await lab.compile(name, target)

    @app.get("/models")
    def models() -> list[dict[str, Any]]:
        """Selectable live models and remaining SYNAPSE-local free-tier budgets."""
        config = load_config("models.yaml")["providers"]
        guard = QuotaGuard(lab.quota_path)
        choices = [{"provider": "mock", "model": None, "label": "Mock · offline",
                    "limits": None, "remaining": None}]
        for provider in ("gemini", "groq"):
            for tier in ("small", "large"):
                model = str(config[provider][tier])
                choices.append({"provider": provider, "model": model,
                                "label": f"{provider.title()} · {model}",
                                "limits": LIMITS[provider].__dict__,
                                "remaining": guard.status(provider, model)})
        return choices

    @app.post("/runs")
    async def start_run(request: RunRequest) -> dict[str, Any]:
        """Start one run; follow it on WS /ws/runs/{run_id}, read it at GET /runs/{run_id}."""
        name, target = lab.target(request.workflow)
        if not target.cases:
            raise HTTPException(422, f"{name} has no test cases to run")
        try:
            case = target.case(request.case_id) if request.case_id else target.cases[0]
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        if request.provider == "mock" and request.live:
            raise HTTPException(422, "mock provider cannot be combined with live=true")
        has_llm = any(node.kind is NodeKind.LLM for node in target.workflow.nodes)
        live = has_llm and (request.live or request.provider in ("gemini", "groq"))
        provider = ((request.provider or load_config("models.yaml")["live_provider"])
                    if live else "mock")
        if request.model is not None and has_llm:
            tiers = load_config("models.yaml")["providers"][provider]
            if not live or request.model not in (tiers["small"], tiers["large"]):
                raise HTTPException(422, "model is not configured for the selected provider")
        plan = None
        if request.mode == "compiled":
            plan = (lab.plan(request.plan_id, name) if request.plan_id
                    else await lab.current_plan(name, target))
            if live:
                plan = lab.live_plan(target, plan, explicit=request.plan_id is not None,
                                     provider=provider, model=request.model)
        scale = (request.scale if request.scale is not None
                 else load_config("mock.yaml").get("latency_scale", 1.0))
        run = lab.start(name, target, case, request.mode, plan, live, scale,
                        provider, request.model if has_llm else None, request.cache_mode)
        return run.meta()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> JSONResponse:
        """The run's record when finished; 202 with its status while it runs."""
        run = lab.runs.get(run_id)
        if run is not None:
            body = lab.run_body(run)
            return JSONResponse(body, status_code=202 if body["status"] == "running" else 200)
        with lab.store() as store:
            try:
                record = store.load_run(run_id)
            except KeyError:
                raise HTTPException(404, f"no run {run_id!r}") from None
        failed_node = next((node.error for node in record.nodes if node.error), None)
        return JSONResponse({**record.model_dump(mode="json"),
                             "status": "failed" if failed_node else "done",
                             "error": failed_node, "usage_totals": usage_totals(record),
                             "cost_unit": cost_unit("gemini" if record.live else "mock")})

    @app.websocket("/ws/runs/{run_id}")
    async def run_events(websocket: WebSocket, run_id: str) -> None:
        """Every event of the run so far, then each new one, ending with run_end."""
        await websocket.accept()
        run = lab.runs.get(run_id)
        try:
            if run is not None:
                async for event in run.follow():
                    await websocket.send_json(event)
            else:
                with lab.store() as store:
                    try:
                        events = replay_events(store.load_run(run_id))
                    except KeyError:
                        events = [{"type": "error", "run_id": run_id,
                                   "message": f"no run {run_id!r}"}]
                for event in events:
                    await websocket.send_json(event)
            await websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            return  # the client left; the run itself carries on

    @app.post("/validate")
    async def validate_route(request: ValidateRequest) -> dict[str, Any]:
        """Validate the workflow's passes on every case; returns checks and verdicts."""
        name, target = lab.target(request.workflow)
        return await lab.validate(name, target, request)

    @app.get("/bench/latest")
    def bench_latest(mode: Literal["mock", "live"] = "mock") -> Any:
        """Measured benchmark JSON, with a bundled mock example on a fresh checkout."""
        path = results_path(mode == "live", lab.results_dir)
        if mode == "mock" and not path.exists() and lab.results_dir == RESULTS_DIR:
            path = ROOT / "bench" / "sample_mock.json"
        if not path.exists():
            raise HTTPException(404, f"no {mode} benchmark yet; run make bench")
        return json.loads(path.read_text())

    return app


app = create_app()
