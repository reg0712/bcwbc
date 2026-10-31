"""SQLite persistence for run traces and the later optimizer's cache and audit log."""

from __future__ import annotations

import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synapse import ROOT
from synapse.ir import ExecPlan, NodeRun, RunRecord, Usage, Workflow

SCHEMA = """
CREATE TABLE runs (run_id TEXT PRIMARY KEY, workflow TEXT, mode TEXT, plan_id TEXT,
  case_id TEXT, live INTEGER, cold_cache INTEGER, wall_ms REAL, created_at TEXT,
  latency_scale REAL);
CREATE TABLE node_runs (run_id TEXT, node_id TEXT, start_s REAL, end_s REAL,
  tokens_in INTEGER, tokens_out INTEGER, model TEXT, cost REAL, cache_hit INTEGER,
  cpu_pct REAL, mem_mb REAL, error TEXT, output_json TEXT, input_hash TEXT,
  retries INTEGER DEFAULT 0, usage_json TEXT);
CREATE TABLE decisions (plan_id TEXT, workflow TEXT, pass_name TEXT, nodes_json TEXT,
  reason TEXT, status TEXT, predicted_ms REAL, predicted_cost REAL,
  measured_ms REAL, measured_cost REAL);
CREATE TABLE cache (key TEXT PRIMARY KEY, entity TEXT, value_json TEXT,
  created_at REAL, ttl_s INTEGER);
CREATE TABLE checks (run_id TEXT, case_id TEXT, node_id TEXT, kind TEXT,
  passed INTEGER, detail TEXT);
CREATE TABLE validation_proofs (plan_id TEXT, workflow TEXT, workflow_hash TEXT,
  cases_json TEXT, backend_key TEXT, passed INTEGER, created_at TEXT,
  PRIMARY KEY (plan_id, workflow_hash, cases_json, backend_key));
"""


def _workflow_hash(workflow: Workflow) -> str:
    """Fingerprint the definition and prompt/schema bytes that were validated."""
    from synapse.passes.memoise import _file_digests
    source = json.dumps({"workflow": workflow.model_dump(mode="json"),
                         "files": {node.id: _file_digests(node) for node in workflow.nodes}},
                        sort_keys=True,
                        separators=(",", ":"))
    return hashlib.sha256(source.encode()).hexdigest()


def _backend_key(provider: str, model_names: dict[str, str] | None = None) -> str:
    """Identify the actual models whose small-tier outputs were checked."""
    if model_names is None:
        from synapse.config import load_config
        tiers = load_config("models.yaml")["providers"][provider]
        model_names = {tier: str(tiers[tier]) for tier in ("small", "large")}
    return json.dumps({"provider": provider, "models": model_names}, sort_keys=True)


class Store:
    """One SQLite database. A run and all its nodes commit as one transaction."""

    def __init__(self, path: str | Path = ROOT / "data" / "synapse.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        with self.connection:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    self.connection.execute(statement.strip().replace("CREATE TABLE ",
                                                                     "CREATE TABLE IF NOT EXISTS ", 1))
            columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(node_runs)")}
            if "input_hash" not in columns:
                self.connection.execute("ALTER TABLE node_runs ADD COLUMN input_hash TEXT")
            if "retries" not in columns:
                self.connection.execute("ALTER TABLE node_runs ADD COLUMN retries INTEGER DEFAULT 0")
            if "usage_json" not in columns:
                self.connection.execute("ALTER TABLE node_runs ADD COLUMN usage_json TEXT")
            run_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(runs)")}
            if "latency_scale" not in run_columns:
                self.connection.execute("ALTER TABLE runs ADD COLUMN latency_scale REAL")

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_: object) -> None:
        self.connection.close()

    def save_run(self, record: RunRecord) -> None:
        """Persist a run atomically; a repeated run id is an error."""
        with self.connection:
            self.connection.execute(
                "INSERT INTO runs (run_id, workflow, mode, plan_id, case_id, live, "
                "cold_cache, wall_ms, created_at, latency_scale) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.run_id, record.workflow, record.mode, record.plan_id, record.case_id,
                 int(record.live), int(record.cold_cache), record.wall_ms,
                 datetime.now(timezone.utc).isoformat(), record.latency_scale))
            self.connection.executemany(
                "INSERT INTO node_runs (run_id, node_id, start_s, end_s, tokens_in, tokens_out, "
                "model, cost, cache_hit, cpu_pct, mem_mb, error, output_json, input_hash, "
                "retries, usage_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(record.run_id, node.node_id, node.start_s, node.end_s,
                  node.usage.tokens_in, node.usage.tokens_out, node.usage.model,
                  node.usage.cost, int(node.cache_hit), node.cpu_pct, node.mem_mb,
                  node.error, json.dumps(node.output, ensure_ascii=False),
                  node.input_hash, node.retries, node.usage.model_dump_json()) for node in record.nodes])

    def load_run(self, run_id: str) -> RunRecord:
        """Read one complete trace; preserve its execution order."""
        row = self.connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        nodes = []
        for item in self.connection.execute(
                "SELECT * FROM node_runs WHERE run_id=? ORDER BY rowid", (run_id,)):
            nodes.append(NodeRun(
                run_id=run_id, node_id=item["node_id"], start_s=item["start_s"],
                end_s=item["end_s"], usage=(Usage.model_validate_json(item["usage_json"])
                if item["usage_json"] else Usage(tokens_in=item["tokens_in"],
                tokens_out=item["tokens_out"], model=item["model"], cost=item["cost"],
                retries=item["retries"] or 0)),
                cache_hit=bool(item["cache_hit"]), cpu_pct=item["cpu_pct"],
                mem_mb=item["mem_mb"], error=item["error"],
                output=json.loads(item["output_json"]), input_hash=item["input_hash"],
                retries=item["retries"] or 0))
        return RunRecord(run_id=run_id, workflow=row["workflow"], mode=row["mode"],
                         plan_id=row["plan_id"], case_id=row["case_id"],
                         live=bool(row["live"]), cold_cache=bool(row["cold_cache"]),
                         latency_scale=row["latency_scale"],
                         wall_ms=row["wall_ms"], nodes=nodes)

    def list_runs(self, workflow: str | None = None) -> list[str]:
        """Return run ids in creation order, optionally for one workflow."""
        query = "SELECT run_id FROM runs"
        params: tuple[Any, ...] = ()
        if workflow is not None:
            query += " WHERE workflow=?"
            params = (workflow,)
        return [row[0] for row in self.connection.execute(query + " ORDER BY rowid", params)]

    def save_decisions(self, plan: ExecPlan) -> None:
        """Persist the compiler audit record for this plan."""
        with self.connection:
            self.connection.execute("DELETE FROM decisions WHERE plan_id=?", (plan.plan_id,))
            self.connection.executemany(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(plan.plan_id, plan.workflow.name, decision.pass_name,
                  json.dumps(decision.nodes), decision.reason, decision.status,
                  decision.predicted_ms, decision.predicted_cost,
                  decision.measured_ms, decision.measured_cost)
                 for decision in plan.decisions])

    def save_validation(self, source: Workflow, plan: ExecPlan,
                        case_ids: list[str], passed: bool, *, provider: str = "mock",
                        model_names: dict[str, str] | None = None) -> None:
        """Record exactly which current workflow and cases approved a plan."""
        cases_json = json.dumps(sorted(set(case_ids)))
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO validation_proofs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (plan.plan_id, source.name, _workflow_hash(source), cases_json,
                 _backend_key(provider, model_names),
                 int(passed), datetime.now(timezone.utc).isoformat()))

    def route_validated(self, workflow: Workflow, *, provider: str = "mock",
                        model_names: dict[str, str] | None = None,
                        plan: ExecPlan | None = None) -> bool:
        """Whether the current routed plan passed every case of the current suite."""
        from synapse.compiler import compile_workflow
        from mock_services import suite_for
        try:
            cases_json = json.dumps(sorted(case["case_id"]
                                          for case in suite_for(workflow.name).cases()))
        except LookupError:
            return False
        candidate = plan or compile_workflow(workflow, passes=("dead_nodes", "cse", "compact_inputs", "parallelise",
                                                        "memoise", "routing"),
                                     routing_probe=True)
        return self.connection.execute(
            "SELECT 1 FROM validation_proofs WHERE plan_id=? AND workflow_hash=? "
            "AND cases_json=? AND backend_key=? AND passed=1 LIMIT 1",
            (candidate.plan_id, _workflow_hash(workflow), cases_json,
             _backend_key(provider, model_names))).fetchone() is not None
