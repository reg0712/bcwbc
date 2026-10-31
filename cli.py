"""SYNAPSE command line: run, compile, profile, compare, validate, bench, import-langgraph.

Mock mode is the default; live model calls need --live plus keys in .env.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import networkx as nx

from mock_services import Suite, suite_for
from synapse.backends import Backends, mock_backends, open_backends
from synapse.config import load_config
from synapse.compiler import compile_workflow
from synapse.executor import execute, final_outputs, resolve
from synapse.graph import GraphError, build_graph, dependencies, levels, write_chain
from synapse.ir import NodeRun, RunRecord, Workflow
from synapse.parser import ParseError, load_workflow

def _add_live(parser: argparse.ArgumentParser) -> None:
    """Add the --live flag: real model calls instead of the mock backends."""
    parser.add_argument("--live", action="store_true",
                        help="call real models (needs keys in .env); mock mode is the default")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one subcommand per verb."""
    parser = argparse.ArgumentParser(
        prog="synapse",
        description="Compile, profile, optimise and validate AI agent workflows.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a workflow on one test case or all of them")
    run.add_argument("workflow", help="workflow YAML file")
    run.add_argument("--mode", choices=["baseline", "compiled"], default="baseline")
    run.add_argument("--case", default="all", help="test case id such as A01, or 'all'")
    run.add_argument("--latency-scale", type=float,
                     help="multiply every mock delay (default: config/mock.yaml)")
    run.add_argument("--bank-url", help="call a running mock bank (make mock) instead of an "
                                        "in-process one, e.g. http://127.0.0.1:8001")
    _add_live(run)

    compile_ = sub.add_parser("compile", help="compile a workflow into an optimised plan")
    compile_.add_argument("workflow", help="workflow YAML file")
    compile_.add_argument("--show-graph", action="store_true",
                          help="print the compiled dependency graph")
    compile_.add_argument("--explain", action="store_true",
                          help="print each pass decision with its reason and predicted gain")

    profile = sub.add_parser("profile", help="profile per-node latency, tokens, cost and "
                                             "the critical path")
    profile.add_argument("workflow", help="workflow YAML file")
    profile.add_argument("--runs", type=int, default=3, help="runs to profile")
    profile.add_argument("--case", default="A01", help="case id to repeat (default: A01)")
    profile.add_argument("--latency-scale", type=float,
                         help="multiply mock delays (default: config/mock.yaml)")
    profile.add_argument("--db", help="SQLite results path (default: data/synapse.db)")
    _add_live(profile)

    compare = sub.add_parser("compare", help="run the original and compiled workflow "
                                             "side by side")
    compare.add_argument("workflow", help="workflow YAML file")
    compare.add_argument("--case", default="A01", help="test case id")
    compare.add_argument("--latency-scale", type=float, help="multiply mock delays")
    _add_live(compare)

    validate = sub.add_parser("validate", help="check compiled outputs against golden runs "
                                               "and reject failing passes")
    validate.add_argument("workflow", help="workflow YAML file")
    validate.add_argument("--unsafe-route-risk", action="store_true",
                          help="add the deliberately unsafe pass: risk summary on the "
                               "small model")
    validate.add_argument("--case", default="all", help="case id or all 20 cases")
    validate.add_argument("--latency-scale", type=float, default=0.01,
                          help="multiply mock delays (default: 0.01 for validation)")
    _add_live(validate)

    bench = sub.add_parser("bench", help="benchmark cold and warm runs with a pass-by-pass "
                                         "ablation")
    bench.add_argument("workflows", nargs="*",
                       help="workflow files (default: every workflow under workflows/)")
    bench.add_argument("--repeats", type=int,
                       help="runs per configuration (default: 10 mock, 3 live)")
    bench.add_argument("--latency-scale", type=float,
                       help="multiply mock delays (default: 0.01 mock, 1 live)")
    bench.add_argument("--cases", help="comma-separated case ids (default: every case)")
    bench.add_argument("--steps", help="comma-separated ablation steps (default: all)")
    bench.add_argument("--output", help="JSON result path (default: bench/results/latest.json)")
    _add_live(bench)

    importer = sub.add_parser("import-langgraph", help="import a LangGraph workflow module "
                                                       "into the SYNAPSE IR")
    importer.add_argument("module", help="module exporting build_graph, NODES, EFFECTS and "
                                         "SAMPLE_INPUTS")
    importer.add_argument("--show-graph", action="store_true",
                          help="print the imported dependency graph")
    importer.add_argument("--run", action="store_true",
                          help="run and validate all sample inputs against the compiled plan")
    return parser


def _not_built(what: str, milestone: str) -> int:
    """Report a command or option that a later milestone delivers."""
    print(f"synapse {what}: not built yet; arrives in {milestone}", file=sys.stderr)
    return 1


def format_graph(g: nx.DiGraph) -> str:
    """One row per node, grouped into levels that could run together, then the write order."""
    rows = [("level", "node", "kind", "effect", "model", "reads from")]
    for number, group in enumerate(levels(g), start=1):
        for node_id in group:
            node = g.nodes[node_id]["node"]
            rows.append((str(number), node_id, node.kind.value, node.effect.value,
                         node.model or "-", ", ".join(dependencies(g, node_id)) or "-"))
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]) - 1)]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)) + "  " + row[-1]
             for row in rows]
    lines.append(f"write order: {' -> '.join(write_chain(g)) or 'none'}")
    return "\n".join(lines)


def cmd_compile(args: argparse.Namespace) -> int:
    """Parse a workflow, build its dependency graph and summarise it."""
    wf = load_workflow(args.workflow)
    if args.explain:
        from synapse.optimiser import optimise
        from synapse.profiler import build_profile
        from synapse.store import Store
        with Store() as store:
            # Only mock baseline runs describe the original workflow's own costs; compiled
            # and live runs would skew every predicted gain.
            records = [record for record in map(store.load_run, store.list_runs(wf.name))
                       if record.mode == "baseline" and not record.live]
            if records:
                latest_scale = records[-1].latency_scale
                records = [record for record in records if record.latency_scale == latest_scale]
            validated_routing = store.route_validated(wf)
        plan = optimise(wf, build_profile(wf, records) if records else None,
                        validated_routing=validated_routing)
    else:
        plan = compile_workflow(wf)
    g = build_graph(plan.workflow)
    print(f"{wf.name}: {len(wf.nodes)} source nodes -> {g.number_of_nodes()} compiled nodes, "
          f"{g.number_of_edges()} data/order edges, {len(levels(g))} levels")
    if args.show_graph:
        print()
        print(format_graph(g))
    if args.explain:
        for decision in plan.decisions:
            gain = (f"; predicted {decision.predicted_ms:.1f} ms" if decision.predicted_ms is not None
                    else "")
            print(f"{decision.pass_name}: {decision.status} {', '.join(decision.nodes) or '-'}; "
                  f"{decision.reason}{gain}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run test cases through a workflow and print each decision and wall time."""
    wf = load_workflow(args.workflow)
    build_graph(wf)  # a bad graph fails before anything runs
    suite = suite_for(wf.name)
    cases = suite.cases() if args.case == "all" else [suite.case(args.case)]
    return asyncio.run(_run_cases(wf, suite, cases, args))


def cmd_compare(args: argparse.Namespace) -> int:
    """Run one case through the original and the compiled workflow and compare them."""
    wf = load_workflow(args.workflow)
    case = suite_for(wf.name).case(args.case)
    return asyncio.run(_compare_case(wf, case, args))


def cmd_validate(args: argparse.Namespace) -> int:
    if args.live and args.unsafe_route_risk:
        raise ValueError("the deliberately unsafe routing demo runs only with mock models")
    wf = load_workflow(args.workflow)
    suite = suite_for(wf.name)
    cases = suite.cases() if args.case == "all" else [suite.case(args.case)]
    return asyncio.run(_validate_cases(wf, cases, args))


def cmd_bench(args: argparse.Namespace) -> int:
    from bench.runner import run_benchmarks, summary_line

    split = lambda value: [part.strip() for part in value.split(",") if part.strip()] if value else None
    results, target = run_benchmarks(
        args.workflows, repeats=args.repeats, scale=args.latency_scale, live=args.live,
        case_ids=split(args.cases), steps=split(args.steps), output=args.output,
        progress=lambda message: print(message, file=sys.stderr, flush=True))
    for result in results:
        print(f"{summary_line(result)}; {target}")
    return 0


def cmd_import_langgraph(args: argparse.Namespace) -> int:
    from synapse.importers.langgraph_importer import import_langgraph
    imported = import_langgraph(args.module)
    wf = imported.workflow
    graph = build_graph(wf)
    print(f"{wf.name}: imported {len(wf.nodes)} nodes, {len(imported.control_edges)} control edges, "
          f"{len(imported.samples)} samples")
    if args.show_graph:
        print(format_graph(graph))
    if args.run:
        return asyncio.run(_run_imported(imported))
    return 0


async def _run_imported(imported) -> int:
    from synapse.backends.retrieval import KeywordRetriever
    from synapse.validator import validate
    wf = imported.workflow
    plan = compile_workflow(wf, passes=("parallelise",))
    plan.order_edges.extend(imported.control_edges)
    async with mock_backends(0, retriever=KeywordRetriever(),
                             functions=imported.functions) as backends:
        golden = [await execute(wf, case, "baseline", backends) for case in imported.samples]
        candidate = [await execute(wf, case, "compiled", backends, plan=plan)
                     for case in imported.samples]
    checks = validate(golden, candidate, 0)
    passed = sum(check.passed for check in checks)
    print(f"validation: {passed}/{len(checks)} checks passed; "
          f"median wall {statistics.median(r.wall_ms for r in golden):.1f} -> "
          f"{statistics.median(r.wall_ms for r in candidate):.1f} ms")
    return 0 if checks and passed == len(checks) else 1


async def _validate_cases(wf: Workflow, cases: list[dict[str, Any]],
                          args: argparse.Namespace) -> int:
    from synapse.validator import validate_workflow
    async with open_backends(args.live, args.latency_scale) as backends:
        await backends.retriever.warm({node.index for node in wf.nodes if node.index})
        plan, checks, floor = await validate_workflow(
            wf, cases, backends, unsafe_route_risk=args.unsafe_route_risk, live=args.live)
        proof_provider, proof_models = backends.provider, backends.model_names.copy()
    passed = sum(check.passed for check in checks)
    print(f"{wf.name}: {len(cases)} cases; {passed}/{len(checks)} checks passed; "
          f"baseline noise floor {floor:.3f}")
    for decision in plan.decisions:
        if decision.status == "rejected":
            print(f"rejected {decision.pass_name}: {decision.reason}")
    print(f"accepted plan: {plan.plan_id}; {len(plan.workflow.nodes)} nodes")
    from synapse.store import Store
    with Store() as store:
        store.save_decisions(plan)
        store.save_validation(wf, plan, [case["case_id"] for case in cases],
                              bool(checks) and all(check.passed for check in checks),
                              provider=proof_provider, model_names=proof_models)
    routed = [f"{node.id}={node.model}" for node in plan.workflow.nodes
              if node.kind.value == "llm"]
    print("models: " + ", ".join(routed))
    return 0 if checks and all(check.passed for check in checks) else 1


async def _compare_case(wf: Workflow, case: dict[str, Any], args: argparse.Namespace) -> int:
    plan = compile_workflow(wf)
    suite = suite_for(wf.name)
    scale = args.latency_scale
    async with open_backends(args.live, scale) as backends:
        await backends.retriever.warm({node.index for node in wf.nodes if node.index})
        print(_backends_label(backends))
        baseline = await execute(wf, case, "baseline", backends, live=args.live)
        compiled = await execute(wf, case, "compiled", backends, plan=plan, live=args.live)
    for record in (baseline, compiled):
        failure = _failure(record)
        if failure:
            print(f"{record.mode}: error at {failure.node_id}: {failure.error}", file=sys.stderr)
            return 1
        retries = sum(n.retries for n in record.nodes)
        print(f"{record.mode}: {record.wall_ms:.1f} ms, "
              f"{_cost_text(sum(n.usage.cost for n in record.nodes), backends.provider)}, "
              f"decision {resolve(suite.key_field, final_outputs(record))}"
              + (f", {retries} retried call{'s' if retries != 1 else ''}" if retries else ""))
    same = resolve(suite.key_field, final_outputs(baseline)) == resolve(
        suite.key_field, final_outputs(compiled))
    print(f"decision match: {same}; speedup: {baseline.wall_ms / compiled.wall_ms:.2f}x")
    return 0 if same else 1


def cmd_profile(args: argparse.Namespace) -> int:
    """Measure repeated baseline runs and save every trace to SQLite."""
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    wf = load_workflow(args.workflow)
    build_graph(wf)
    case = suite_for(wf.name).case(args.case)
    return asyncio.run(_profile_runs(wf, case, args))


async def _profile_runs(wf: Workflow, case: dict[str, Any], args: argparse.Namespace) -> int:
    from synapse.profiler import build_profile
    from synapse.store import Store

    scale = args.latency_scale
    records: list[RunRecord] = []
    async with open_backends(args.live, scale) as backends:
        await backends.retriever.warm({node.index for node in wf.nodes if node.index})
        with Store(args.db) if args.db else Store() as store:
            for _ in range(args.runs):
                record = await execute(wf, case, "baseline", backends, live=args.live)
                store.save_run(record)
                records.append(record)
    provider = backends.provider
    profile = build_profile(wf, records)
    unit = "mock units" if provider == "mock" else "USD est."
    print(f"{wf.name}: {len(records)} baseline runs, case {args.case}; {_backends_label(backends)}")
    print(f"node                   p50 ms   p95 ms  tokens in/out  {'cost (' + unit + ')':>17}  repeat")
    for node in wf.nodes:
        stat = profile.nodes.get(node.id)
        if stat is None:
            continue
        cost = f"{stat.cost:17.3f}" if provider == "mock" else f"{stat.cost:17.6f}"
        print(f"{node.id:<22} {stat.p50_ms:7.1f} {stat.p95_ms:8.1f} "
              f"{stat.tokens_in:6.0f}/{stat.tokens_out:<6.0f} "
              f"{cost} {stat.repeat_input_rate:7.0%}")
    print(f"critical path: {' -> '.join(profile.critical_path)}")
    mean_cost = statistics.mean(sum(n.usage.cost for n in r.nodes) for r in records)
    print(f"median wall: {statistics.median(r.wall_ms for r in records):.1f} ms; "
          f"mean cost: {_cost_text(mean_cost, provider)}")
    failures = [(r.case_id, n.node_id, n.error) for r in records for n in r.nodes if n.error]
    for case_id, node_id, error in failures:
        print(f"{case_id}: error at {node_id}: {error}", file=sys.stderr)
    return 1 if failures else 0


async def _run_cases(wf: Workflow, suite: Suite, cases: list[dict[str, Any]],
                     args: argparse.Namespace) -> int:
    """Run cases one after another on shared mock backends, printing each as it finishes."""
    scale = args.latency_scale
    if scale is None:
        scale = load_config("mock.yaml").get("latency_scale", 1.0)
    records: list[RunRecord] = []
    plan = compile_workflow(wf) if args.mode == "compiled" else None
    from synapse.passes.memoise import ExactCache
    cache = ExactCache() if plan else None
    async with open_backends(args.live, scale, bank_url=args.bank_url) as backends:
        await backends.retriever.warm({node.index for node in wf.nodes if node.index})
        header = f"{wf.name}: {args.mode} run, {_backends_label(backends)}"
        if scale != 1:
            header += f", latency scale {scale:g}"
        if args.bank_url:
            header += f", bank at {args.bank_url}"
        print(header + "\n", flush=True)
        if len(cases) > 1:
            print(RUN_COLUMNS, flush=True)
        for case in cases:
            record = await execute(wf, case, args.mode, backends, plan=plan, cache=cache,
                                   live=args.live)
            records.append(record)
            if len(cases) == 1:
                print(format_timeline(record) + "\n\n" + RUN_COLUMNS)
            print(_run_row(record, suite), flush=True)
    if len(cases) > 1:
        print("\n" + _run_summary(records, suite))
    return 0 if all(_failure(record) is None for record in records) else 1


RUN_COLUMNS = "case  decision  wall time"


def _backends_label(backends: Backends) -> str:
    """Which backends a run uses: all mock, or a live model with the mock bank."""
    if backends.provider == "mock":
        return "mock backends"
    return f"live {backends.provider} models, mock bank and local retrieval"


def _cost_text(cost: float, provider: str) -> str:
    """Model cost: mock units, or US dollars estimated at the provider's paid-tier prices."""
    if provider == "mock":
        return f"{cost:.3f} mock cost units"
    return f"${cost:.6f} estimated at paid-tier prices"


def _failure(record: RunRecord) -> NodeRun | None:
    """The node a run stopped at, if one failed."""
    return next((node for node in record.nodes if node.error is not None), None)


def _run_row(record: RunRecord, suite: Suite) -> str:
    """One case: its decision and wall time, or the node it failed at."""
    failure = _failure(record)
    if failure is not None:
        return f"{record.case_id:<4}  error at {failure.node_id}: {failure.error}"
    decision = resolve(suite.key_field, final_outputs(record))
    return f"{record.case_id:<4}  {str(decision):<8}  {record.wall_ms / 1000:7.2f} s"


def _run_summary(records: list[RunRecord], suite: Suite) -> str:
    """Decision counts, agreement with the test rule and median wall time."""
    finished = [record for record in records if _failure(record) is None]
    decisions = {record.case_id: resolve(suite.key_field, final_outputs(record))
                 for record in finished}
    counts = ", ".join(f"{n} {decision}" for decision, n in Counter(decisions.values()).most_common())
    matches = sum(decision == suite.expected(case_id) for case_id, decision in decisions.items())
    summary = (f"{len(records)} cases: {counts}; {matches} of {len(records)} match the test "
               f"rule; median wall time {statistics.median(r.wall_ms for r in records) / 1000:.2f} s")
    failed = len(records) - len(finished)
    return summary + (f"; {failed} failed" if failed else "")


def format_timeline(record: RunRecord) -> str:
    """When each node of one run started and ended, in seconds from the start of the run,
    with retries and cache hits noted."""
    rows = [("node", "start s", "end s", "")] + [
        (node.node_id, f"{node.start_s:.2f}", f"{node.end_s:.2f}", _node_notes(node))
        for node in record.nodes]
    width = max(len(row[0]) for row in rows)
    return "\n".join(f"{name:<{width}}  {start:>7}  {end:>7}  {notes}".rstrip()
                     for name, start, end, notes in rows)


def _node_notes(node: NodeRun) -> str:
    """Retries and cache hits, the things that explain an unusual bar length."""
    notes = []
    if node.retries:
        notes.append(f"{node.retries} retr{'y' if node.retries == 1 else 'ies'}")
    if node.cache_hit:
        notes.append("cache hit")
    return ", ".join(notes)


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": cmd_run,
    "compile": cmd_compile,
    "profile": cmd_profile,
    "compare": cmd_compare,
    "validate": cmd_validate,
    "bench": cmd_bench,
    "import-langgraph": cmd_import_langgraph,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, dispatch to the chosen command, and return the exit code."""
    args = build_parser().parse_args(argv)
    handler = HANDLERS.get(args.command)
    if handler is None:
        return _not_built(args.command, "a later milestone")
    try:
        return handler(args)
    except (OSError, LookupError, ValueError, ParseError, GraphError) as exc:
        print(f"synapse {args.command}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
