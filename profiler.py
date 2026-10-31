"""Resource measurements and profile summaries from actual run records."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence

import psutil

from synapse.graph import build_graph, critical_path
from synapse.ir import NodeRun, NodeStats, Profile, RunRecord, Workflow

_PROCESS = psutil.Process()


def resource_snapshot() -> float:
    """Process CPU seconds before a node starts."""
    times = _PROCESS.cpu_times()
    return times.user + times.system


def resource_usage(cpu_before: float, elapsed_s: float) -> tuple[float, float]:
    """Process CPU percent over elapsed wall time and current RSS in MiB."""
    cpu = max(0.0, (resource_snapshot() - cpu_before) / max(elapsed_s, 1e-9) * 100)
    return cpu, _PROCESS.memory_info().rss / (1024 * 1024)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def build_profile(wf: Workflow, records: Sequence[RunRecord]) -> Profile:
    """Aggregate measured nodes and find the graph's latency-weighted critical path."""
    if not records:
        raise ValueError("a profile needs at least one run")
    if any(record.workflow != wf.name for record in records):
        raise ValueError("profile records must belong to the same workflow")
    by_node: dict[str, list[NodeRun]] = defaultdict(list)
    for record in records:
        for node in record.nodes:
            by_node[node.node_id].append(node)
    stats: dict[str, NodeStats] = {}
    for node_id, samples in by_node.items():
        durations = [(sample.end_s - sample.start_s) * 1000 for sample in samples]
        seen: set[str] = set()
        repeats = 0
        for sample in samples:
            if sample.input_hash is not None:
                repeats += sample.input_hash in seen
                seen.add(sample.input_hash)
        stats[node_id] = NodeStats(
            node_id=node_id, p50_ms=statistics.median(durations),
            p95_ms=_percentile(durations, 0.95),
            tokens_in=statistics.mean(s.usage.tokens_in for s in samples),
            tokens_out=statistics.mean(s.usage.tokens_out for s in samples),
            cost=statistics.mean(s.usage.cost for s in samples),
            repeat_input_rate=repeats / len(samples),
        )
    path, _ = critical_path(build_graph(wf), {name: stat.p50_ms for name, stat in stats.items()})
    return Profile(workflow=wf.name, runs=len(records), nodes=stats, critical_path=path)
