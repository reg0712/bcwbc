"""Cold and warm ablation benchmarks: one executor, one test set, one concurrency cap and the
same inputs for every configuration; only the compiler passes differ.

Method (build spec): the headline is the cache-cold median wall time over repeated runs of
every test case; warm-cache runs ("repeat customer") are reported separately. A repeat runs
every configuration over every case, so slow drift on the machine hits all configurations
alike. Every cold run starts from an empty cache. A warm configuration keeps one cache, filled
by an unmeasured pass over the cases before measuring. Warm runs are measured only where a
cache can matter: the baseline (which has none) and the configuration with caching.

Mock latencies are scaled by DEFAULT_SCALE. At 0.001 (the old default) fixed Python and
embedding overhead was most of every run; at 0.01 the configured delays are about 85 to 90%
of measured wall time and a full benchmark takes a few minutes. Live runs default to the
configured delays (scale 1) because their model latency is real.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from synapse import ROOT
from synapse.backends import open_backends
from synapse.compiler import compile_workflow
from synapse.config import load_config
from synapse.executor import execute, final_outputs, resolve
from synapse.graph import build_graph, critical_path
from synapse.ir import ExecPlan, NodeKind, RunRecord, Workflow
from synapse.passes.memoise import ExactCache

DEFAULT_SCALE = 0.01
MIN_REPEATS = {"mock": 10, "live": 3}
RESULTS_DIR = ROOT / "bench" / "results"
WORKFLOWS_DIR = ROOT / "workflows"


@dataclass(frozen=True)
class Step:
    """One ablation configuration: the passes applied on top of the baseline."""

    name: str
    label: str
    passes: tuple[str, ...]


STEPS: tuple[Step, ...] = (
    Step("baseline", "baseline", ()),
    Step("dead_nodes", "+ dead nodes", ("dead_nodes",)),
    Step("cse", "+ duplicates (CSE)", ("dead_nodes", "cse")),
    Step("compact_inputs", "+ compact inputs", ("dead_nodes", "cse", "compact_inputs")),
    Step("parallelise", "+ parallelisation", ("dead_nodes", "cse", "compact_inputs", "parallelise")),
    Step("routing", "+ routing", ("dead_nodes", "cse", "compact_inputs", "parallelise", "routing")),
    Step("memoise", "+ caching", ("dead_nodes", "cse", "compact_inputs", "parallelise", "routing", "memoise")),
)


@dataclass
class Target:
    """A workflow with what it needs to run: test cases, imported functions for LangGraph
    nodes, and the decision field with the value each case should reach (when known)."""

    workflow: Workflow
    cases: list[dict[str, Any]]
    functions: dict[str, Callable[..., Any]] = field(default_factory=dict)
    key_field: str | None = None
    expected: Callable[[str], Any] | None = None
    source: str = ""

    def case(self, case_id: str) -> dict[str, Any]:
        """One case by id; LookupError names the valid ids."""
        for case in self.cases:
            if str(case.get("case_id")) == case_id:
                return case
        ids = ", ".join(str(case.get("case_id")) for case in self.cases)
        raise LookupError(f"no case {case_id!r} for {self.workflow.name}; cases are {ids}")


def target_for(wf: Workflow, *, functions: Mapping[str, Callable[..., Any]] | None = None,
               samples: Sequence[dict[str, Any]] = (), source: str = "") -> Target:
    """A Target from the workflow's test suite, or from sample inputs when it has none."""
    from mock_services import suite_for

    try:
        suite = suite_for(wf.name)
    except LookupError:  # no suite: the samples are the cases (there may be none)
        cases = [{**sample, "case_id": str(sample.get("case_id", f"S{i:02d}"))}
                 for i, sample in enumerate(samples, start=1)]
        return Target(wf, cases, dict(functions or {}), source=source)
    return Target(wf, suite.cases(), dict(functions or {}), suite.key_field, suite.expected,
                  source=source)


def load_target(path: str | Path) -> Target:
    """Load a workflow file: a YAML workflow, or a LangGraph module through the importer."""
    path = Path(path)
    if path.suffix == ".py":
        from synapse.importers.langgraph_importer import import_langgraph

        imported = import_langgraph(str(path))
        return target_for(imported.workflow, functions=imported.functions,
                          samples=imported.samples, source=str(path))
    from synapse.parser import load_workflow

    return target_for(load_workflow(path), source=str(path))


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile of a non-empty sample."""
    values = sorted(samples)
    point = (len(values) - 1) * fraction
    low = int(point)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (point - low)


def cost_unit(provider: str) -> str:
    """What a cost figure means for a provider."""
    return "mock units" if provider == "mock" else "USD (estimated at paid-tier prices)"


def results_path(live: bool = False, directory: Path = RESULTS_DIR) -> Path:
    """Where the latest results go: mock and live are reported separately."""
    return directory / ("latest_live.json" if live else "latest.json")


async def benchmark(target: Target | Workflow, repeats: int | None = None,
                    scale: float | None = None, case_ids: Sequence[str] | None = None, *,
                    live: bool = False, steps: Sequence[str] | None = None,
                    backend_options: Mapping[str, Any] | None = None,
                    progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Measure every ablation step, cold and (where a cache exists) warm; returns the result
    document written to bench/results/. Defaults: 10 repeats and DEFAULT_SCALE in mock mode,
    3 repeats at the configured delays live."""
    if isinstance(target, Workflow):
        target = target_for(target)
    mode = "live" if live else "mock"
    repeats = MIN_REPEATS[mode] if repeats is None else repeats
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if scale is None:
        scale = DEFAULT_SCALE if not live else load_config("mock.yaml").get("latency_scale", 1.0)
    chosen = _choose_steps(steps)
    cases = target.cases if case_ids is None else [target.case(id) for id in case_ids]
    if not cases:
        raise ValueError("a benchmark needs at least one case")
    wf = target.workflow
    plans = {step.name: compile_workflow(wf, passes=step.passes,
                                         routing_probe="routing" in step.passes)
             if step.passes else None
             for step in chosen}
    warm = [step for step in chosen if step.name == "baseline" or "memoise" in step.passes]
    say = progress or (lambda message: None)
    started = time.perf_counter()
    records: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    options = dict(backend_options or {})
    if target.functions:
        options.setdefault("functions", target.functions)

    async with open_backends(live, scale, **options) as backends:
        await backends.retriever.warm({node.index for node in wf.nodes if node.index})

        async def run(step: Step, case: dict[str, Any], cache: ExactCache) -> RunRecord:
            plan = plans.get(step.name)
            return await execute(wf, case, "compiled" if plan else "baseline", backends,
                                 plan=plan, cache=cache, live=live)

        for step in chosen:  # one discarded run each pays one-time costs (imports, schemas)
            await run(step, cases[0], ExactCache())
        reference = {}
        if target.expected is None:  # no decision rule: outputs must match the original
            reference = {str(case["case_id"]): _outputs(wf, await run(STEPS[0], case,
                                                                      ExactCache()))
                         for case in cases}
        warm_caches = {step.name: ExactCache() for step in warm}
        for step in warm:  # the repeat customer: each case has been seen once before
            if "memoise" in step.passes:
                for case in cases:
                    await run(step, case, warm_caches[step.name])
        for repeat in range(1, repeats + 1):
            for step in chosen:
                for cache_mode in ("cold", "warm") if step in warm else ("cold",):
                    for case in cases:
                        cache = ExactCache() if cache_mode == "cold" else warm_caches[step.name]
                        records[step.name, cache_mode].append(await run(step, case, cache))
            say(f"{wf.name} {mode}: repeat {repeat}/{repeats} done after "
                f"{time.perf_counter() - started:.0f} s")
        provider = getattr(backends, "provider", "mock")
        cap = getattr(backends, "max_concurrency", None)

    configurations = [
        _summarise(step, cache_mode, wf, plans[step.name], records[step.name, cache_mode],
                   target, reference)
        for step in chosen for cache_mode in ("cold", "warm") if (step.name, cache_mode) in records]
    return {"workflow": wf.name, "source": target.source, "mode": mode, "provider": provider,
            "cost_unit": cost_unit(provider),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.perf_counter() - started, 1),
            "repeats": repeats, "cases": len(cases),
            "case_ids": [str(case["case_id"]) for case in cases],
            "latency_scale": scale, "concurrency_cap": cap,
            "correctness": (f"{target.key_field} matches the test rule" if target.expected
                            else "outputs match the original workflow"),
            "method": ("median and p95 wall time over repeats x cases; cold runs start with "
                       "an empty cache, warm runs reuse a cache filled by one unmeasured pass"),
            "steps": [{"name": s.name, "label": s.label, "passes": list(s.passes)}
                      for s in chosen],
            "configurations": configurations,
            "summary": _headline(configurations)}


def _choose_steps(names: Sequence[str] | None) -> list[Step]:
    """The ablation steps to run, always in ablation order."""
    if names is None:
        return list(STEPS)
    unknown = sorted(set(names) - {step.name for step in STEPS})
    if unknown:
        raise ValueError(f"unknown benchmark step(s): {', '.join(unknown)}; steps are "
                         f"{', '.join(step.name for step in STEPS)}")
    return [step for step in STEPS if step.name in names]


def _outputs(wf: Workflow, record: RunRecord) -> dict[str, Any]:
    """The declared outputs of a run, or everything it produced when none are declared."""
    produced = final_outputs(record)
    if not wf.outputs:
        return produced
    return {field: produced.get(field.split(".", 1)[0]) for field in wf.outputs}


def _correct(record: RunRecord, target: Target, reference: Mapping[str, Any]) -> bool:
    """A run is correct when it finished and reached the expected decision (or, without a
    decision rule, the original workflow's outputs)."""
    if any(node.error for node in record.nodes):
        return False
    if target.expected is None or target.key_field is None:
        return _outputs(target.workflow, record) == reference.get(record.case_id)
    try:
        return resolve(target.key_field, final_outputs(record)) == target.expected(record.case_id)
    except KeyError:
        return False


def _schedule(wf: Workflow, plan: ExecPlan | None) -> nx.DiGraph:
    """The graph the executor actually follows: file order for the baseline and for plans
    without the parallel pass, the dependency graph plus write barriers otherwise."""
    active = plan.workflow if plan else wf
    if plan is None or "parallelise" not in {d.pass_name for d in plan.decisions
                                             if d.status == "applied"}:
        chain = nx.DiGraph()
        chain.add_nodes_from((node.id, {"index": i}) for i, node in enumerate(active.nodes))
        chain.add_edges_from((a.id, b.id) for a, b in zip(active.nodes, active.nodes[1:]))
        return chain
    graph = build_graph(active)
    graph.add_edges_from(plan.order_edges)
    return graph


def _summarise(step: Step, cache_mode: str, wf: Workflow, plan: ExecPlan | None,
               records: list[RunRecord], target: Target,
               reference: Mapping[str, Any]) -> dict[str, Any]:
    """Every benchmark metric for one configuration's runs."""
    kinds = {node.id: node.kind for node in (plan.workflow if plan else wf).nodes}
    durations = [record.wall_ms for record in records]
    per_node: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for node in record.nodes:
            per_node[node.node_id].append((node.end_s - node.start_s) * 1000)
    path, path_ms = critical_path(_schedule(wf, plan),
                                  {id: statistics.median(ms) for id, ms in per_node.items()})

    def mean(count: Callable[[RunRecord], float]) -> float:
        return statistics.mean(count(record) for record in records)

    def calls(record: RunRecord, wanted: set[NodeKind]) -> int:
        return sum(1 for node in record.nodes if kinds.get(node.node_id) in wanted
                   and not node.cache_hit and node.error is None)

    def model_calls(record: RunRecord) -> int:  # an escalation to the large model is a call
        return sum(1 + max(0, node.retries - getattr(node.usage, "retries", 0))
                   for node in record.nodes
                   if kinds.get(node.node_id) is NodeKind.LLM and not node.cache_hit)

    correct = sum(_correct(record, target, reference) for record in records)
    return {
        "name": step.name, "label": step.label, "passes": list(step.passes),
        "cache": cache_mode, "runs": len(records),
        "p50_ms": statistics.median(durations), "p95_ms": percentile(durations, 0.95),
        "mean_ms": statistics.mean(durations),
        "critical_path_ms": path_ms, "critical_path": path,
        "mean_cost": mean(lambda r: sum(node.usage.cost for node in r.nodes)),
        "mean_llm_calls": mean(model_calls),
        "mean_tool_calls": mean(lambda r: calls(r, {NodeKind.TOOL, NodeKind.RETRIEVE})),
        "mean_rate_limit_retries": mean(
            lambda r: sum(getattr(node.usage, "retries", 0) for node in r.nodes)),
        "mean_tokens_in": mean(lambda r: sum(node.usage.tokens_in for node in r.nodes)),
        "mean_tokens_out": mean(lambda r: sum(node.usage.tokens_out for node in r.nodes)),
        "mean_cache_hits": mean(lambda r: sum(node.cache_hit for node in r.nodes)),
        "correct": correct, "correct_rate": correct / len(records),
        "errors": sum(any(node.error for node in record.nodes) for record in records),
    }


def _headline(configurations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Baseline against the final configuration, cold (headline) and warm."""
    def find(name: str, cache: str) -> Mapping[str, Any] | None:
        return next((c for c in configurations if c["name"] == name and c["cache"] == cache),
                    None)

    cold = [c for c in configurations if c["cache"] == "cold"]
    if not cold:
        return {}
    first, last = cold[0], cold[-1]
    summary: dict[str, Any] = {
        "from": first["name"], "to": last["name"],
        "cold_p50_ms": [first["p50_ms"], last["p50_ms"]],
        "cold_speedup": first["p50_ms"] / last["p50_ms"] if last["p50_ms"] else None,
        "mean_cost": [first["mean_cost"], last["mean_cost"]],
        "mean_llm_calls": [first["mean_llm_calls"], last["mean_llm_calls"]],
        "correct_rate": [first["correct_rate"], last["correct_rate"]],
    }
    warm_first, warm_last = find(first["name"], "warm"), find(last["name"], "warm")
    if warm_first and warm_last:
        summary["warm_p50_ms"] = [warm_first["p50_ms"], warm_last["p50_ms"]]
        summary["warm_speedup"] = (warm_first["p50_ms"] / warm_last["p50_ms"]
                                   if warm_last["p50_ms"] else None)
    return summary


def summary_line(result: Mapping[str, Any]) -> str:
    """One line for the terminal: cold and warm medians, speedup and correctness."""
    head = result.get("summary") or {}
    if not head:
        return f"{result['workflow']}: no configurations measured"
    before, after = head["cold_p50_ms"]
    line = (f"{result['workflow']} ({result['mode']}, scale {result['latency_scale']:g}): "
            f"{result['cases']} cases x {result['repeats']} repeats; cold p50 "
            f"{before:.1f} -> {after:.1f} ms ({head['cold_speedup']:.2f}x)")
    if "warm_p50_ms" in head:
        line += f"; warm p50 {head['warm_p50_ms'][0]:.1f} -> {head['warm_p50_ms'][1]:.1f} ms"
    return line + f"; correct {head['correct_rate'][1]:.0%} after"


def write_results(results: Sequence[Mapping[str, Any]], live: bool = False,
                  output: str | Path | None = None) -> Path:
    """Write results to bench/results/latest.json (latest_live.json for live runs), plus a
    timestamped copy beside it so earlier results stay comparable."""
    target = Path(output) if output else results_path(live)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(list(results), indent=2)
    target.write_text(text)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        (target.parent / f"{'live' if live else 'mock'}-{stamp}.json").write_text(text)
    return target


def run_benchmarks(paths: Sequence[str | Path] | None = None, *, repeats: int | None = None,
                   scale: float | None = None, live: bool = False,
                   case_ids: Sequence[str] | None = None, steps: Sequence[str] | None = None,
                   output: str | Path | None = None,
                   progress: Callable[[str], None] | None = None
                   ) -> tuple[list[dict[str, Any]], Path]:
    """Benchmark workflow files (default: the loan workflow) and write the results."""
    targets = [load_target(path) for path in (paths or [WORKFLOWS_DIR / "loan_prescreen.yaml"])]
    results = [asyncio.run(benchmark(target, repeats, scale, case_ids, live=live, steps=steps,
                                     progress=progress)) for target in targets]
    return results, write_results(results, live, output)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m bench.runner`: the same benchmark as `synapse bench`."""
    parser = argparse.ArgumentParser(prog="bench.runner", description=__doc__.split("\n")[0])
    parser.add_argument("workflows", nargs="*", help="workflow files (default: loan_prescreen)")
    parser.add_argument("--repeats", type=int, help="repeats per configuration "
                                                    "(default: 10 mock, 3 live)")
    parser.add_argument("--latency-scale", type=float, help=f"multiply mock delays (default: "
                                                            f"{DEFAULT_SCALE} mock, 1 live)")
    parser.add_argument("--cases", help="comma-separated case ids (default: every case)")
    parser.add_argument("--steps", help="comma-separated ablation steps (default: all)")
    parser.add_argument("--output", help="JSON path (default: bench/results/latest.json)")
    parser.add_argument("--live", action="store_true", help="call real models")
    args = parser.parse_args(argv)
    split = lambda text: [part.strip() for part in text.split(",") if part.strip()] if text else None
    results, target = run_benchmarks(
        args.workflows, repeats=args.repeats, scale=args.latency_scale, live=args.live,
        case_ids=split(args.cases), steps=split(args.steps), output=args.output,
        progress=lambda message: print(message, file=sys.stderr, flush=True))
    for result in results:
        print(f"{summary_line(result)}; {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
