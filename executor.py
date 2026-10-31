"""One async executor for both modes, so every speedup comes from the graph rewrite.

Baseline mode awaits nodes in file order, the way most hand-written agents run. Compiled mode
launches every node whose inputs are ready, capped by a semaphore per provider (M5).
"""

from __future__ import annotations

import asyncio
import time
import uuid
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from synapse.backends import Backends, open_backends
from synapse.backends.llm import output_schema, render_prompt
from synapse.backends.llm import InvalidStructuredOutput, LLM_EVENTS
from synapse.ir import ExecPlan, Node, NodeKind, NodeRun, RunRecord, Usage, Workflow
from synapse.cost import estimate_cost
from synapse.graph import build_graph
from synapse.ir import Effect
from synapse.passes.memoise import ExactCache, cache_key

NodeResult = tuple[dict[str, Any], Usage, int]


async def run(wf: Workflow, case: dict[str, Any], mode: Literal["baseline", "compiled"],
              plan: ExecPlan | None = None, live: bool = False,
              cache: ExactCache | None = None,
              on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None) -> RunRecord:
    """Run one test case on fresh backends (live LLM calls when live is set). Callers running
    many cases should open the backends once and call execute()."""
    async with open_backends(live=live) as backends:
        return await execute(wf, case, mode, backends, plan=plan, live=live,
                             cache=cache, on_event=on_event)


async def execute(wf: Workflow, case: dict[str, Any], mode: Literal["baseline", "compiled"],
                  backends: Backends, plan: ExecPlan | None = None,
                  live: bool = False, cache: ExactCache | None = None,
                  on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None) -> RunRecord:
    """Run one test case through the workflow on the given backends."""
    if mode == "compiled" and plan is None:
        from synapse.compiler import compile_workflow
        plan = compile_workflow(wf)
    active = plan.workflow if mode == "compiled" and plan else wf
    missing = [name for name in wf.inputs if name not in case]
    if missing:
        raise ValueError(f"case {case.get('case_id')!r} lacks workflow input(s): "
                         f"{', '.join(missing)}")
    run_id = uuid.uuid4().hex[:12]
    cold_cache = not (cache is not None and cache.entries)
    state = {name: case[name] for name in wf.inputs}
    started = time.perf_counter()
    nodes: list[NodeRun] = []
    if mode == "baseline":
        for node in wf.nodes:
            record = await _run_node(node, state, backends, run_id, started,
                                     on_event=on_event)
            nodes.append(record)
            if record.error is not None:
                break
            state.update(record.output)
    elif plan is not None and not any(
            decision.pass_name == "parallelise" and decision.status == "applied"
            for decision in plan.decisions):
        cache = cache if cache is not None else ExactCache()
        for node in active.nodes:
            record = await _run_node(node, state, backends, run_id, started,
                                     cache=cache if node.id in plan.cached_nodes else None,
                                     on_event=on_event)
            nodes.append(record)
            if record.error is not None:
                break
            state.update(record.output)
            if node.effect is Effect.WRITE:
                cache.invalidate(None)
    else:
        assert plan is not None
        cache = cache if cache is not None else ExactCache()
        graph = build_graph(active)
        graph.add_edges_from(plan.order_edges)
        semaphore = asyncio.Semaphore(backends.max_concurrency)  # the provider's cap

        async def launch(node: Node, snapshot: Mapping[str, Any]) -> NodeRun:
            if node.kind is NodeKind.LLM:
                async with semaphore:
                    return await _run_node(node, snapshot, backends, run_id, started,
                                           cache=cache if node.id in plan.cached_nodes else None,
                                           on_event=on_event)
            return await _run_node(node, snapshot, backends, run_id, started,
                                   cache=cache if node.id in plan.cached_nodes else None,
                                   on_event=on_event)

        pending = set(graph.nodes)
        completed: set[str] = set()
        running: dict[asyncio.Task[NodeRun], str] = {}
        failed = False
        while pending or running:
            if not failed:
                ready = sorted((id for id in pending if set(graph.predecessors(id)) <= completed),
                               key=lambda id: graph.nodes[id]["index"])
                for id in ready:
                    pending.remove(id)
                    task = asyncio.create_task(launch(graph.nodes[id]["node"], state.copy()))
                    running[task] = id
            if not running:
                break
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                id = running.pop(task)
                record = task.result()
                nodes.append(record)
                if record.error:
                    failed = True
                else:
                    state.update(record.output)
                    completed.add(id)
                    if graph.nodes[id]["node"].effect is Effect.WRITE:
                        cache.invalidate(None)
    return RunRecord(run_id=run_id, workflow=wf.name, mode=mode,
                     plan_id=plan.plan_id if plan else None, case_id=str(case.get("case_id", "")),
                     live=live, cold_cache=cold_cache, latency_scale=backends.latency_scale,
                     wall_ms=(time.perf_counter() - started) * 1000,
                     nodes=nodes)


def final_outputs(record: RunRecord) -> dict[str, Any]:
    """Every output field a run produced, merged."""
    merged: dict[str, Any] = {}
    for node in record.nodes:
        merged.update(node.output)
    return merged


def resolve(reference: str, state: Mapping[str, Any]) -> Any:
    """The value an input reference names: 'applicant.pan' reads state['applicant']['pan']."""
    root, *path = reference.split(".")
    if root not in state:
        raise KeyError(f"{reference}: {root!r} has not been produced")
    value = state[root]
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"{reference}: no field {key!r}")
        value = value[key]
    return value


async def _run_node(node: Node, state: Mapping[str, Any], backends: Backends, run_id: str,
                    started: float, cache: ExactCache | None = None,
                    on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None) -> NodeRun:
    """Run one node and time it; a failure is recorded on the NodeRun, not raised."""
    start_s = time.perf_counter() - started
    output: dict[str, Any] = {}
    usage, error = Usage(), None
    retries = 0
    input_hash = None
    cache_hit = False
    if on_event is not None:
        await on_event({"type": "node_start", "node_id": node.id, "at_s": start_s})
    from synapse.profiler import resource_snapshot, resource_usage
    before = resource_snapshot()
    async def model_event(event: dict[str, Any]) -> None:
        if on_event is not None:
            await on_event({**event, "at_s": time.perf_counter() - started})
    event_token = LLM_EVENTS.set(model_event)
    try:
        inputs = {reference: resolve(reference, state) for reference in node.inputs}
        canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
        input_hash = hashlib.sha256(canonical.encode()).hexdigest()
        settings = {"provider": backends.provider,
                    "model_name": backends.model_names.get(str(node.model))}
        if node.kind is not NodeKind.LLM:
            settings["backend"] = backends.cache_namespace
        key = cache_key(node, inputs, settings=settings) if cache is not None else None
        cached = cache.get(key) if cache is not None and key is not None else None
        if cached is not None:
            output, cache_hit = cached, True
        else:
            output, usage, escalations = await _RUNNERS[node.kind](node, inputs, backends)
            retries = escalations + usage.retries
            if usage.cost == 0:
                usage.cost = estimate_cost(usage, node.model, provider=backends.provider)
            if cache is not None and key is not None:
                cache.put(key, output, node.ttl_s if node.effect is Effect.READ else None,
                          None, read=node.effect is Effect.READ)
    except Exception as exc:  # the run stops at this node and reports why
        if isinstance(getattr(exc, "usage", None), Usage):
            usage = exc.usage
            if usage.cost == 0:
                usage.cost = estimate_cost(usage, node.model, provider=backends.provider)
            retries = usage.retries
        error = f"{type(exc).__name__}: {exc}"
    finally:
        LLM_EVENTS.reset(event_token)
    end_s = time.perf_counter() - started
    cpu_pct, mem_mb = resource_usage(before, end_s - start_s)
    record = NodeRun(run_id=run_id, node_id=node.id, start_s=start_s,
                     end_s=end_s, usage=usage, output=output, error=error,
                     input_hash=input_hash, cpu_pct=cpu_pct, mem_mb=mem_mb,
                     cache_hit=cache_hit, retries=retries)
    if on_event is not None:
        await on_event({"type": "node_finish", "node_id": node.id, "at_s": end_s,
                        "error": error, "cache_hit": cache_hit,
                        "usage": usage.model_dump(mode="json")})
    return record


def _assign(node: Node, value: Any) -> dict[str, Any]:
    """A node's output fields: the whole value for one output, else one key per field."""
    if len(node.outputs) <= 1:
        return {field: value for field in node.outputs}
    if not isinstance(value, Mapping) or any(field not in value for field in node.outputs):
        raise ValueError(f"{node.id}: result lacks output fields {node.outputs}")
    return {field: value[field] for field in node.outputs}


async def _run_llm(node: Node, inputs: dict[str, Any], backends: Backends) -> NodeResult:
    schema = output_schema(node)
    prompt = render_prompt(node, inputs)
    try:
        result, usage = await backends.llm.invoke(node, prompt, schema)
        return _assign(node, result if schema is not None else result["text"]), usage, 0
    except Exception as first_error:
        from pydantic import ValidationError
        if node.fallback_model is None or not isinstance(
                first_error, (ValidationError, InvalidStructuredOutput)):
            raise
        fallback = node.model_copy(update={"model": node.fallback_model,
                                           "fallback_model": None})
        small_usage = (first_error.usage if isinstance(first_error, InvalidStructuredOutput)
                       else Usage(model=backends.model_names.get(str(node.model)), usage_complete=False))
        try:
            result, large_usage = await backends.llm.invoke(fallback, prompt, schema)
        except Exception as second_error:
            large_usage = getattr(second_error, "usage", Usage(
                model=backends.model_names.get(node.fallback_model), usage_complete=False))
            combined = _combine_usage(small_usage, large_usage, node, backends)
            error = RuntimeError(f"human review required: both model tiers failed: {second_error}")
            error.usage = combined
            raise error from second_error
        usage = _combine_usage(small_usage, large_usage, node, backends)
        return _assign(node, result if schema is not None else result["text"]), usage, 1


def _combine_usage(small: Usage, large: Usage, node: Node, backends: Backends) -> Usage:
    """Retain both calls' known usage, even when both schemas fail."""
    combined = large.model_copy(deep=True)
    combined.cost = (small.cost or estimate_cost(small, node.model, provider=backends.provider)) + (
        large.cost or estimate_cost(large, node.fallback_model, provider=backends.provider))
    for field in ("tokens_in", "tokens_out", "retries", "calls", "quota_wait_ms",
                  "retry_wait_ms", "provider_ms"):
        setattr(combined, field, getattr(large, field) + getattr(small, field))
    combined.usage_complete = small.usage_complete and large.usage_complete
    combined.model = f"{small.model}+{large.model}"
    return combined


async def _run_tool(node: Node, inputs: dict[str, Any], backends: Backends) -> NodeResult:
    return _assign(node, await backends.tools.call(str(node.tool), inputs)), Usage(), 0


async def _run_retrieve(node: Node, inputs: dict[str, Any], backends: Backends) -> NodeResult:
    query = node.query or str(next(iter(inputs.values()), ""))
    return _assign(node, await backends.retriever.search(str(node.index), query)), Usage(), 0


async def _run_transform(node: Node, inputs: dict[str, Any], backends: Backends) -> NodeResult:
    function = backends.functions.get(node.tool or node.id)
    if function is None:
        raise LookupError(f"no imported function for {node.id}")
    if asyncio.iscoroutinefunction(function):
        result = await function(inputs)
    else:
        result = await asyncio.to_thread(function, inputs)
    if not isinstance(result, Mapping) or any(field not in result for field in node.outputs):
        raise ValueError(f"{node.id}: imported function did not return declared outputs")
    return {field: result[field] for field in node.outputs}, Usage(), 0


async def _not_runnable(node: Node, inputs: dict[str, Any], backends: Backends) -> NodeResult:
    raise NotImplementedError(f"{node.kind.value} nodes cannot run from YAML yet")


_RUNNERS: dict[NodeKind, Callable[[Node, dict[str, Any], Backends], Awaitable[NodeResult]]] = {
    NodeKind.LLM: _run_llm,
    NodeKind.TOOL: _run_tool,
    NodeKind.RETRIEVE: _run_retrieve,
    NodeKind.TRANSFORM: _run_transform,
    NodeKind.GATE: _not_runnable,
}
